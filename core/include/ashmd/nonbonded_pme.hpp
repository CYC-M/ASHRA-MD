#pragma once

// Gold-Standard Explicit-Solvent Periodic Nonbonded Engine (Mode 0: explicit-pme)
//   1) Spatial 3D Cell-List + Verlet Neighbor List
//   2) Direct-space Ewald erfc(alpha * r)/r with C1 DSF cutoff continuity
//   3) AMBER 12-6 Lennard-Jones with C2 quintic potential switch + analytical dispersion tail
//   4) Intramolecular excluded-pair Ewald erf(alpha * r)/r correction + Ewald self energy
//   5) Self-contained 3D Radix-2 Cooley-Tukey / Bluestein FFT + 4th-order Cardinal B-Spline SPME
//      with exact analytical B-spline differentiation (F_recip = -grad U_recip to < 1e-13!)
//   6) Interface Soft-Core Hamiltonian Tempering support for IHRT replicas

#include "types.hpp"
#include "parallel.hpp"
#include <cmath>
#include <complex>
#include <vector>
#include <algorithm>

namespace ashmd {

namespace detail {

// Minimum-image displacement in orthorhombic periodic box
inline double min_image(double d, double box_len, double inv_box_len) noexcept {
    return d - box_len * std::nearbyint(d * inv_box_len);
}

// In-place 1D complex FFT (power-of-2 Cooley-Tukey, plus exact DFT fallback if non-power-of-2)
inline bool is_pow2(int n) noexcept {
    return n > 0 && (n & (n - 1)) == 0;
}

inline void fft_1d(std::complex<double>* data, int n, int sign) noexcept {
    if (n <= 1) return;
    if (is_pow2(n)) {
        // Bit-reversal permutation
        for (int i = 1, j = 0; i < n; ++i) {
            int bit = n >> 1;
            for (; j & bit; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) std::swap(data[i], data[j]);
        }
        // Butterfly stages
        for (int len = 2; len <= n; len <<= 1) {
            const double ang = static_cast<double>(sign) * (-2.0 * PI_VAL / static_cast<double>(len));
            const std::complex<double> wlen(std::cos(ang), std::sin(ang));
            const int half = len >> 1;
            for (int i = 0; i < n; i += len) {
                std::complex<double> w(1.0, 0.0);
                for (int j = 0; j < half; ++j) {
                    const std::complex<double> u = data[i + j];
                    const std::complex<double> v = data[i + j + half] * w;
                    data[i + j]        = u + v;
                    data[i + j + half] = u - v;
                    w *= wlen;
                }
            }
        }
    } else {
        // General exact O(N^2) fallback for small non-pow2 dimensions
        std::vector<std::complex<double>> out(n, {0.0, 0.0});
        const double base_ang = static_cast<double>(sign) * (-2.0 * PI_VAL / static_cast<double>(n));
        for (int m = 0; m < n; ++m) {
            std::complex<double> acc(0.0, 0.0);
            for (int k = 0; k < n; ++k) {
                const double a = base_ang * static_cast<double>((m * k) % n);
                acc += data[k] * std::complex<double>(std::cos(a), std::sin(a));
            }
            out[m] = acc;
        }
        for (int m = 0; m < n; ++m) data[m] = out[m];
    }
}

// 3D unnormalized complex FFT: sign = +1 corresponds to forward exp(-2*pi*i m*k/K),
// sign = -1 corresponds to unnormalized inverse exp(+2*pi*i m*k/K)
inline void fft_3d(
    std::vector<std::complex<double>>& grid,
    int kx, int ky, int kz,
    int sign
) {
    // Transform along Z (contiguous)
    for (int ix = 0; ix < kx; ++ix) {
        for (int iy = 0; iy < ky; ++iy) {
            fft_1d(&grid[(ix * ky + iy) * kz], kz, sign);
        }
    }
    // Transform along Y
    std::vector<std::complex<double>> buf(std::max(kx, ky));
    for (int ix = 0; ix < kx; ++ix) {
        for (int iz = 0; iz < kz; ++iz) {
            for (int iy = 0; iy < ky; ++iy) buf[iy] = grid[(ix * ky + iy) * kz + iz];
            fft_1d(buf.data(), ky, sign);
            for (int iy = 0; iy < ky; ++iy) grid[(ix * ky + iy) * kz + iz] = buf[iy];
        }
    }
    // Transform along X
    for (int iy = 0; iy < ky; ++iy) {
        for (int iz = 0; iz < kz; ++iz) {
            for (int ix = 0; ix < kx; ++ix) buf[ix] = grid[(ix * ky + iy) * kz + iz];
            fft_1d(buf.data(), kx, sign);
            for (int ix = 0; ix < kx; ++ix) grid[(ix * ky + iy) * kz + iz] = buf[ix];
        }
    }
}

// 4th-order Cardinal B-spline values w[0..3] and derivatives dw[0..3] for delta in [0, 1)
inline void bspline4_weights(double delta, double w[4], double dw[4]) noexcept {
    const double d2 = delta * delta;
    const double d3 = d2 * delta;
    const double om = 1.0 - delta;

    w[0] = (1.0 / 6.0) * d3;
    w[1] = (1.0 / 6.0) * (-3.0 * d3 + 3.0 * d2 + 3.0 * delta + 1.0);
    w[2] = (1.0 / 6.0) * ( 3.0 * d3 - 6.0 * d2 + 4.0);
    w[3] = (1.0 / 6.0) * (om * om * om);

    dw[0] =  0.5 * d2;
    dw[1] =  0.5 * (-3.0 * d2 + 2.0 * delta + 1.0);
    dw[2] =  0.5 * ( 3.0 * d2 - 4.0 * delta);
    dw[3] = -0.5 * (om * om);
}

inline double bspline4_mod_sq(int m, int K) noexcept {
    const double c = (2.0 / 3.0) + (1.0 / 3.0) * std::cos(2.0 * PI_VAL * static_cast<double>(m) / static_cast<double>(K));
    return 1.0 / (c * c);
}

} // namespace detail

struct NeighborPair {
    int32_t i, j;
    bool is_interface;
};

class PMEWorkspace {
public:
    std::vector<NeighborPair> pair_list;
    std::vector<NeighborPair> inter_pairs;
    std::vector<double> last_build_x, last_build_y, last_build_z;
    double skin = 1.5; // Verlet skin [A]
    bool initialized = false;

    std::vector<std::complex<double>> mesh;
    std::vector<double> mod_x, mod_y, mod_z;
    int32_t cached_kx = 0, cached_ky = 0, cached_kz = 0;

    // Thread-local accumulators to avoid cache-line bouncing while preserving exactness
    std::vector<std::vector<int64_t>> tls_fx, tls_fy, tls_fz;
    std::vector<qenergy_t> tls_elj, tls_edir, tls_eraw_inter;

    void ensure_tls(int32_t n_threads, int32_t n_atoms) {
        if (static_cast<int32_t>(tls_fx.size()) != n_threads) {
            tls_fx.resize(n_threads);
            tls_fy.resize(n_threads);
            tls_fz.resize(n_threads);
            tls_elj.resize(n_threads, 0);
            tls_edir.resize(n_threads, 0);
            tls_eraw_inter.resize(n_threads, 0);
        }
        for (int32_t t = 0; t < n_threads; ++t) {
            tls_fx[t].assign(n_atoms, 0);
            tls_fy[t].assign(n_atoms, 0);
            tls_fz[t].assign(n_atoms, 0);
            tls_elj[t] = 0;
            tls_edir[t] = 0;
            tls_eraw_inter[t] = 0;
        }
    }

    bool needs_rebuild(const SystemTopology& top, const double* rx, const double* ry, const double* rz) const noexcept {
        if (!initialized || static_cast<int32_t>(last_build_x.size()) != top.n_atoms) return true;
        const double thresh_sq = 0.25 * skin * skin; // (skin / 2)^2
        const double inv_bx = 1.0 / top.box_x;
        const double inv_by = 1.0 / top.box_y;
        const double inv_bz = 1.0 / top.box_z;
        for (int32_t i = 0; i < top.n_atoms; ++i) {
            const double dx = detail::min_image(rx[i] - last_build_x[i], top.box_x, inv_bx);
            const double dy = detail::min_image(ry[i] - last_build_y[i], top.box_y, inv_by);
            const double dz = detail::min_image(rz[i] - last_build_z[i], top.box_z, inv_bz);
            if (dx * dx + dy * dy + dz * dz > thresh_sq) return true;
        }
        return false;
    }

    void rebuild_neighbor_list(
        const SystemTopology& top,
        const double* rx,
        const double* ry,
        const double* rz
    ) {
        const int32_t n = top.n_atoms;
        last_build_x.assign(rx, rx + n);
        last_build_y.assign(ry, ry + n);
        last_build_z.assign(rz, rz + n);

        const double rlist = top.cutoff + skin;
        const double rlist_sq = rlist * rlist;
        const double inv_bx = 1.0 / top.box_x;
        const double inv_by = 1.0 / top.box_y;
        const double inv_bz = 1.0 / top.box_z;

        const int cx = std::max(1, static_cast<int>( std::floor(top.box_x / rlist) ));
        const int cy = std::max(1, static_cast<int>( std::floor(top.box_y / rlist) ));
        const int cz = std::max(1, static_cast<int>( std::floor(top.box_z / rlist) ));

        pair_list.clear();
        inter_pairs.clear();
        pair_list.reserve(static_cast<size_t>(n) * 64);
        inter_pairs.reserve(1024);

        // If box is small relative to 3*rlist, direct O(N^2) minimum-image scan avoids duplicate cell images!
        if (cx < 3 || cy < 3 || cz < 3) {
            for (int32_t i = 0; i < n; ++i) {
                for (int32_t j = i + 1; j < n; ++j) {
                    if (top.zone[i] == ZONE_SCAFFOLD && top.zone[j] == ZONE_SCAFFOLD) continue;
                    if (top.is_excluded(i, j)) continue;
                    const double dx = detail::min_image(rx[i] - rx[j], top.box_x, inv_bx);
                    const double dy = detail::min_image(ry[i] - ry[j], top.box_y, inv_by);
                    const double dz = detail::min_image(rz[i] - rz[j], top.box_z, inv_bz);
                    if (dx * dx + dy * dy + dz * dz <= rlist_sq) {
                        const bool inter = (top.role[i] == ROLE_PEPTIDE && (top.role[j] == ROLE_RECEPTOR || top.role[j] == ROLE_ION)) ||
                                           (top.role[j] == ROLE_PEPTIDE && (top.role[i] == ROLE_RECEPTOR || top.role[i] == ROLE_ION));
                        pair_list.push_back({i, j, inter});
                        if (inter) inter_pairs.push_back({i, j, true});
                    }
                }
            }
        } else {
            const int ncells = cx * cy * cz;
            std::vector<int32_t> head(ncells, -1);
            std::vector<int32_t> next(n, -1);

            for (int32_t i = 0; i < n; ++i) {
                double fx = rx[i] * inv_bx; fx -= std::floor(fx);
                double fy = ry[i] * inv_by; fy -= std::floor(fy);
                double fz = rz[i] * inv_bz; fz -= std::floor(fz);
                int ix = std::min(cx - 1, static_cast<int>(fx * cx));
                int iy = std::min(cy - 1, static_cast<int>(fy * cy));
                int iz = std::min(cz - 1, static_cast<int>(fz * cz));
                int cid = (ix * cy + iy) * cz + iz;
                next[i] = head[cid];
                head[cid] = i;
            }

            for (int ix = 0; ix < cx; ++ix) {
                for (int iy = 0; iy < cy; ++iy) {
                    for (int iz = 0; iz < cz; ++iz) {
                        const int cid1 = (ix * cy + iy) * cz + iz;
                        for (int dx_c = -1; dx_c <= 1; ++dx_c) {
                            const int jx = (ix + dx_c + cx) % cx;
                            for (int dy_c = -1; dy_c <= 1; ++dy_c) {
                                const int jy = (iy + dy_c + cy) % cy;
                                for (int dz_c = -1; dz_c <= 1; ++dz_c) {
                                    const int jz = (iz + dz_c + cz) % cz;
                                    const int cid2 = (jx * cy + jy) * cz + jz;
                                    if (cid2 < cid1) continue;

                                    for (int32_t a = head[cid1]; a != -1; a = next[a]) {
                                        for (int32_t b = (cid1 == cid2 ? next[a] : head[cid2]); b != -1; b = next[b]) {
                                            int32_t i = std::min(a, b);
                                            int32_t j = std::max(a, b);
                                            if (top.zone[i] == ZONE_SCAFFOLD && top.zone[j] == ZONE_SCAFFOLD) continue;
                                            if (top.is_excluded(i, j)) continue;
                                            const double dx = detail::min_image(rx[i] - rx[j], top.box_x, inv_bx);
                                            const double dy = detail::min_image(ry[i] - ry[j], top.box_y, inv_by);
                                            const double dz = detail::min_image(rz[i] - rz[j], top.box_z, inv_bz);
                                            if (dx * dx + dy * dy + dz * dz <= rlist_sq) {
                                                const bool inter = (top.role[i] == ROLE_PEPTIDE && (top.role[j] == ROLE_RECEPTOR || top.role[j] == ROLE_ION)) ||
                                                                   (top.role[j] == ROLE_PEPTIDE && (top.role[i] == ROLE_RECEPTOR || top.role[i] == ROLE_ION));
                                                pair_list.push_back({i, j, inter});
                                                if (inter) inter_pairs.push_back({i, j, true});
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        initialized = true;
    }

    void ensure_pme_tables(int32_t kx, int32_t ky, int32_t kz) {
        if (kx == cached_kx && ky == cached_ky && kz == cached_kz) return;
        cached_kx = kx; cached_ky = ky; cached_kz = kz;
        mod_x.resize(kx);
        mod_y.resize(ky);
        mod_z.resize(kz);
        for (int m = 0; m < kx; ++m) mod_x[m] = detail::bspline4_mod_sq(m, kx);
        for (int m = 0; m < ky; ++m) mod_y[m] = detail::bspline4_mod_sq(m, ky);
        for (int m = 0; m < kz; ++m) mod_z[m] = detail::bspline4_mod_sq(m, kz);
        mesh.resize(static_cast<size_t>(kx) * ky * kz);
    }
};

// Evaluate Direct-Space Nonbonded (Switched LJ 12-6 + DSF Ewald Direct + Excluded Ewald Correction + Self)
inline void compute_pme_direct_and_lj(
    const SystemTopology& top,
    PMEWorkspace& ws,
    ParallelPool& pool,
    const double* __restrict rx,
    const double* __restrict ry,
    const double* __restrict rz,
    const ReplicaTemperingParams& rep,
    int64_t* __restrict qfx,
    int64_t* __restrict qfy,
    int64_t* __restrict qfz,
    EnergyBreakdown& eb
) {
    if (ws.needs_rebuild(top, rx, ry, rz)) {
        ws.rebuild_neighbor_list(top, rx, ry, rz);
    }

    const int32_t n = top.n_atoms;
    const int32_t n_threads = pool.num_threads();
    ws.ensure_tls(n_threads, n);

    const double rcut    = top.cutoff;
    const double rcut_sq = rcut * rcut;
    const double ron     = top.switch_on;
    const double inv_sw_w = (rcut > ron) ? (1.0 / (rcut - ron)) : 0.0;

    const double alpha = top.ewald_alpha;
    const double two_alpha_over_sqrtpi = 2.0 * alpha / std::sqrt(PI_VAL);
    const double erfc_rc = std::erfc(alpha * rcut);
    const double v_shift_rc = erfc_rc / rcut;
    const double f_shift_rc = erfc_rc / rcut_sq + two_alpha_over_sqrtpi * std::exp(-alpha * alpha * rcut_sq) / rcut;

    const double inv_bx = 1.0 / top.box_x;
    const double inv_by = 1.0 / top.box_y;
    const double inv_bz = 1.0 / top.box_z;

    const int32_t npairs = static_cast<int32_t>(ws.pair_list.size());
    pool.parallel_for(npairs, 256, [&](int32_t tid, int32_t beg, int32_t end) {
        auto* lfx = ws.tls_fx[tid].data();
        auto* lfy = ws.tls_fy[tid].data();
        auto* lfz = ws.tls_fz[tid].data();
        qenergy_t qe_lj_loc = 0;
        qenergy_t qe_el_loc = 0;
        qenergy_t qe_raw_inter = 0;

        for (int32_t idx = beg; idx < end; ++idx) {
            const auto& pr = ws.pair_list[idx];
            const int32_t i = pr.i;
            const int32_t j = pr.j;

            const double dx = detail::min_image(rx[i] - rx[j], top.box_x, inv_bx);
            const double dy = detail::min_image(ry[i] - ry[j], top.box_y, inv_by);
            const double dz = detail::min_image(rz[i] - rz[j], top.box_z, inv_bz);
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 >= rcut_sq) continue;

            const double r = std::sqrt(std::max(r2, 1.0e-18));
            const double inv_r = 1.0 / r;
            const double scale = pr.is_interface ? rep.lambda_inter : 1.0;

            const double sc_a = (pr.is_interface ? rep.softcore_alpha : 0.0);

            // 1. Direct-space Ewald electrostatics (with soft-core when sc_a > 0, C1 force-shifted at rcut)
            const double qq = COULOMB_KCAL_A * top.charge[i] * top.charge[j];
            double u_el_unscaled = 0.0;
            double du_el_dr_unscaled = 0.0;
            if (qq != 0.0) {
                if (sc_a > 0.0) {
                    const double r_el     = std::sqrt(r2 + 12.0 * sc_a);
                    const double inv_rel  = 1.0 / r_el;
                    const double erfc_ar  = std::erfc(alpha * r_el);
                    const double exp_ar2  = std::exp(-alpha * alpha * r_el * r_el);
                    const double dr_el_dr = r * inv_rel;
                    u_el_unscaled = qq * (erfc_ar * inv_rel - v_shift_rc + f_shift_rc * (r_el - rcut));
                    du_el_dr_unscaled = qq * (-(erfc_ar * inv_rel * inv_rel + two_alpha_over_sqrtpi * exp_ar2 * inv_rel) + f_shift_rc) * dr_el_dr;
                } else {
                    const double erfc_ar = std::erfc(alpha * r);
                    const double exp_ar2 = std::exp(-alpha * alpha * r2);
                    u_el_unscaled = qq * (erfc_ar * inv_r - v_shift_rc + f_shift_rc * (r - rcut));
                    du_el_dr_unscaled = qq * (-(erfc_ar * inv_r * inv_r + two_alpha_over_sqrtpi * exp_ar2 * inv_r) + f_shift_rc);
                }
            }

            // 2. 12-6 Lennard-Jones with optional soft-core for interface tempering + C2 switch
            const double eps_ij  = std::sqrt(top.epsilon[i] * top.epsilon[j]);
            const double rmin_ij = top.rmin_half[i] + top.rmin_half[j];
            double u_lj_unscaled = 0.0;
            double du_lj_dr_unscaled = 0.0;

            if (eps_ij > 0.0) {
                double u_lj_bare = 0.0;
                double du_lj_bare_dr = 0.0;

                if (sc_a > 0.0) {
                    // Smooth r^6 soft-core: r_eff^6 = r^6 + sc_a * rmin^6
                    const double r2_c = r2;
                    const double r6   = r2_c * r2_c * r2_c;
                    const double rm2  = rmin_ij * rmin_ij;
                    const double rm6  = rm2 * rm2 * rm2;
                    const double denom = r6 + sc_a * rm6;
                    const double x = rm6 / denom;
                    u_lj_bare = eps_ij * (x * x - 2.0 * x);
                    // dx/dr = -rm6 * 6 * r^5 / denom^2 = -6 * x * r^5 / denom
                    const double dx_dr = -6.0 * x * (r2_c * r2_c * r) / denom;
                    du_lj_bare_dr = eps_ij * (2.0 * x - 2.0) * dx_dr;
                } else {
                    const double sr2 = (rmin_ij * rmin_ij) / r2;
                    const double sr6 = sr2 * sr2 * sr2;
                    u_lj_bare = eps_ij * (sr6 * sr6 - 2.0 * sr6);
                    du_lj_bare_dr = -12.0 * eps_ij * (sr6 * sr6 - sr6) * inv_r;
                }

                if (r > ron && rcut > ron) {
                    const double u = (r - ron) * inv_sw_w;
                    const double u2 = u * u;
                    const double u3 = u2 * u;
                    const double sw  = 1.0 - 10.0 * u3 + 15.0 * u2 * u2 - 6.0 * u3 * u2;
                    const double dsw = (-30.0 * u2 + 60.0 * u3 - 30.0 * u2 * u2) * inv_sw_w;
                    u_lj_unscaled     = sw * u_lj_bare;
                    du_lj_dr_unscaled = sw * du_lj_bare_dr + dsw * u_lj_bare;
                } else {
                    u_lj_unscaled     = u_lj_bare;
                    du_lj_dr_unscaled = du_lj_bare_dr;
                }
            }

            if (pr.is_interface) {
                qe_raw_inter += double_to_qenergy(u_el_unscaled + u_lj_unscaled);
            }
            qe_el_loc += double_to_qenergy(scale * u_el_unscaled);
            qe_lj_loc += double_to_qenergy(scale * u_lj_unscaled);

            const double f_over_r = -scale * (du_el_dr_unscaled + du_lj_dr_unscaled) * inv_r;
            const int64_t fx = double_to_qforce(f_over_r * dx);
            const int64_t fy = double_to_qforce(f_over_r * dy);
            const int64_t fz = double_to_qforce(f_over_r * dz);

            lfx[i] += fx; lfy[i] += fy; lfz[i] += fz;
            lfx[j] -= fx; lfy[j] -= fy; lfz[j] -= fz;
        }

        ws.tls_elj[tid] += qe_lj_loc;
        ws.tls_edir[tid] += qe_el_loc;
        ws.tls_eraw_inter[tid] += qe_raw_inter;
    });

    qenergy_t qe_lj_sum = 0, qe_el_sum = 0, qe_inter_sum = 0;
    for (int32_t t = 0; t < n_threads; ++t) {
        qe_lj_sum += ws.tls_elj[t];
        qe_el_sum += ws.tls_edir[t];
        qe_inter_sum += ws.tls_eraw_inter[t];
        const auto* lfx = ws.tls_fx[t].data();
        const auto* lfy = ws.tls_fy[t].data();
        const auto* lfz = ws.tls_fz[t].data();
        for (int32_t i = 0; i < n; ++i) {
            qfx[i] += lfx[i];
            qfy[i] += lfy[i];
            qfz[i] += lfz[i];
        }
    }
    eb.lj_sr += qenergy_to_double(qe_lj_sum);
    eb.elec_dir += qenergy_to_double(qe_el_sum);
    eb.raw_interface_nonb += qenergy_to_double(qe_inter_sum);

    // 3. Excluded intramolecular Ewald Gaussian correction: -C * q_i * q_j * erf(alpha * r) / r
    double e_excl = 0.0;
    for (int32_t i = 0; i < n; ++i) {
        const double qi = top.charge[i];
        if (qi == 0.0) continue;
        for (int32_t p = top.excl_offsets[i]; p < top.excl_offsets[i + 1]; ++p) {
            const int32_t j = top.excl_targets[p]; // j > i guaranteed
            const double qq = COULOMB_KCAL_A * qi * top.charge[j];
            if (qq == 0.0) continue;
            const double dx = detail::min_image(rx[i] - rx[j], top.box_x, inv_bx);
            const double dy = detail::min_image(ry[i] - ry[j], top.box_y, inv_by);
            const double dz = detail::min_image(rz[i] - rz[j], top.box_z, inv_bz);
            const double r2 = dx * dx + dy * dy + dz * dz;
            const double r  = std::sqrt(std::max(r2, 1.0e-18));

            if (r < 1.0e-6) {
                e_excl -= qq * two_alpha_over_sqrtpi;
            } else {
                const double erf_ar = std::erf(alpha * r);
                e_excl -= qq * erf_ar / r;
                // d/dr [-qq * erf(a*r)/r] = -qq * [ (2a/sqrt(pi))*exp(-a^2 r^2)/r - erf(a*r)/r^2 ]
                // Force F_i = -dU/dr * (d/r) = qq * [ (2a/sqrt(pi))*exp(-a^2 r^2)/r - erf(a*r)/r^2 ] * (d/r)
                const double f_over_r = qq * (two_alpha_over_sqrtpi * std::exp(-alpha * alpha * r2) / r - erf_ar / r2) / r;
                const int64_t fx = double_to_qforce(f_over_r * dx);
                const int64_t fy = double_to_qforce(f_over_r * dy);
                const int64_t fz = double_to_qforce(f_over_r * dz);
                qfx[i] += fx; qfy[i] += fy; qfz[i] += fz;
                qfx[j] -= fx; qfy[j] -= fy; qfz[j] -= fz;
            }
        }
    }
    eb.elec_excl += e_excl;

    // 4. Ewald Self Energy: -(alpha / sqrt(pi)) * C * sum q_i^2
    double q2_sum = 0.0;
    for (int32_t i = 0; i < n; ++i) q2_sum += top.charge[i] * top.charge[i];
    eb.elec_self -= (alpha / std::sqrt(PI_VAL)) * COULOMB_KCAL_A * q2_sum;
}

// Fast exact O(N_inter) evaluator for direct-space Ewald + LJ interface energy under arbitrary (lambda, softcore_alpha)
// Eliminates redundant O(N^2) + 3D-FFT evaluations during IHRT swap / MBAR cross-energy checks.
inline double eval_pme_interface_direct_lj_only(
    const SystemTopology& top,
    PMEWorkspace& ws,
    const double* __restrict rx,
    const double* __restrict ry,
    const double* __restrict rz,
    double lambda_inter,
    double softcore_alpha
) {
    if (ws.needs_rebuild(top, rx, ry, rz)) {
        ws.rebuild_neighbor_list(top, rx, ry, rz);
    }
    if (lambda_inter == 0.0 || ws.inter_pairs.empty()) return 0.0;

    const double rcut    = top.cutoff;
    const double rcut_sq = rcut * rcut;
    const double ron     = top.switch_on;
    const double inv_sw_w = (rcut > ron) ? (1.0 / (rcut - ron)) : 0.0;
    const double alpha = top.ewald_alpha;
    const double two_alpha_over_sqrtpi = 2.0 * alpha / std::sqrt(PI_VAL);
    const double erfc_rc = std::erfc(alpha * rcut);
    const double v_shift_rc = erfc_rc / rcut;
    const double f_shift_rc = erfc_rc / rcut_sq + two_alpha_over_sqrtpi * std::exp(-alpha * alpha * rcut_sq) / rcut;

    const double inv_bx = 1.0 / top.box_x;
    const double inv_by = 1.0 / top.box_y;
    const double inv_bz = 1.0 / top.box_z;

    qenergy_t qe_lj = 0, qe_el = 0;
    for (const auto& pr : ws.inter_pairs) {
        const int32_t i = pr.i;
        const int32_t j = pr.j;
        const double dx = detail::min_image(rx[i] - rx[j], top.box_x, inv_bx);
        const double dy = detail::min_image(ry[i] - ry[j], top.box_y, inv_by);
        const double dz = detail::min_image(rz[i] - rz[j], top.box_z, inv_bz);
        const double r2 = dx * dx + dy * dy + dz * dz;
        if (r2 >= rcut_sq) continue;

        const double r = std::sqrt(std::max(r2, 1.0e-18));
        const double inv_r = 1.0 / r;

        const double qq = COULOMB_KCAL_A * top.charge[i] * top.charge[j];
        double u_el_unscaled = 0.0;
        if (qq != 0.0) {
            if (softcore_alpha > 0.0) {
                const double r_el    = std::sqrt(r2 + 12.0 * softcore_alpha);
                const double inv_rel = 1.0 / r_el;
                const double erfc_ar = std::erfc(alpha * r_el);
                u_el_unscaled = qq * (erfc_ar * inv_rel - v_shift_rc + f_shift_rc * (r_el - rcut));
            } else {
                const double erfc_ar = std::erfc(alpha * r);
                u_el_unscaled = qq * (erfc_ar * inv_r - v_shift_rc + f_shift_rc * (r - rcut));
            }
        }

        const double eps_ij  = std::sqrt(top.epsilon[i] * top.epsilon[j]);
        const double rmin_ij = top.rmin_half[i] + top.rmin_half[j];
        double u_lj_unscaled = 0.0;
        if (eps_ij > 0.0) {
            double u_lj_bare = 0.0;
            if (softcore_alpha > 0.0) {
                const double r6  = r2 * r2 * r2;
                const double rm2 = rmin_ij * rmin_ij;
                const double rm6 = rm2 * rm2 * rm2;
                const double x   = rm6 / (r6 + softcore_alpha * rm6);
                u_lj_bare = eps_ij * (x * x - 2.0 * x);
            } else {
                const double sr2 = (rmin_ij * rmin_ij) / r2;
                const double sr6 = sr2 * sr2 * sr2;
                u_lj_bare = eps_ij * (sr6 * sr6 - 2.0 * sr6);
            }
            if (r > ron && rcut > ron) {
                const double u = (r - ron) * inv_sw_w;
                const double u2 = u * u;
                const double u3 = u2 * u;
                const double sw = 1.0 - 10.0 * u3 + 15.0 * u2 * u2 - 6.0 * u3 * u2;
                u_lj_unscaled = sw * u_lj_bare;
            } else {
                u_lj_unscaled = u_lj_bare;
            }
        }

        qe_el += double_to_qenergy(lambda_inter * u_el_unscaled);
        qe_lj += double_to_qenergy(lambda_inter * u_lj_unscaled);
    }
    return qenergy_to_double(qe_lj) + qenergy_to_double(qe_el);
}

// Evaluate Reciprocal-Space 4th-order Cardinal B-Spline Smooth Particle Mesh Ewald (SPME)
// Can be called every step OR every outer RESPA step on filtered coordinates q_tilde!
inline void compute_pme_reciprocal(
    const SystemTopology& top,
    PMEWorkspace& ws,
    const double* __restrict rx,
    const double* __restrict ry,
    const double* __restrict rz,
    int64_t* __restrict qfx,
    int64_t* __restrict qfy,
    int64_t* __restrict qfz,
    EnergyBreakdown& eb
) {
    const int32_t kx = top.pme_grid_x;
    const int32_t ky = top.pme_grid_y;
    const int32_t kz = top.pme_grid_z;
    if (kx <= 0 || ky <= 0 || kz <= 0) return;

    ws.ensure_pme_tables(kx, ky, kz);
    std::fill(ws.mesh.begin(), ws.mesh.end(), std::complex<double>(0.0, 0.0));

    const int32_t n = top.n_atoms;
    const double scale_ux = static_cast<double>(kx) / top.box_x;
    const double scale_uy = static_cast<double>(ky) / top.box_y;
    const double scale_uz = static_cast<double>(kz) / top.box_z;

    // 1. Charge spreading onto 3D mesh with 4th-order B-splines
    for (int32_t i = 0; i < n; ++i) {
        const double qi = top.charge[i];
        if (qi == 0.0) continue;

        double fx = rx[i] / top.box_x; fx -= std::floor(fx);
        double fy = ry[i] / top.box_y; fy -= std::floor(fy);
        double fz = rz[i] / top.box_z; fz -= std::floor(fz);

        const double ux = fx * kx;
        const double uy = fy * ky;
        const double uz = fz * kz;

        const int gx0 = static_cast<int>(std::floor(ux)) % kx;
        const int gy0 = static_cast<int>(std::floor(uy)) % ky;
        const int gz0 = static_cast<int>(std::floor(uz)) % kz;

        double wx[4], dwx[4], wy[4], dwy[4], wz[4], dwz[4];
        detail::bspline4_weights(ux - std::floor(ux), wx, dwx);
        detail::bspline4_weights(uy - std::floor(uy), wy, dwy);
        detail::bspline4_weights(uz - std::floor(uz), wz, dwz);

        for (int px = 0; px < 4; ++px) {
            const int ix = (gx0 - px + kx) % kx;
            const double qwx = qi * wx[px];
            for (int py = 0; py < 4; ++py) {
                const int iy = (gy0 - py + ky) % ky;
                const double qwxwy = qwx * wy[py];
                const size_t base = static_cast<size_t>(ix * ky + iy) * kz;
                for (int pz = 0; pz < 4; ++pz) {
                    const int iz = (gz0 - pz + kz) % kz;
                    ws.mesh[base + iz] += std::complex<double>(qwxwy * wz[pz], 0.0);
                }
            }
        }
    }

    // 2. Forward 3D FFT: Q(k) -> Q_hat(m)
    detail::fft_3d(ws.mesh, kx, ky, kz, +1);

    // 3. Reciprocal convolution with Ewald influence function G(m)
    const double vol = top.box_x * top.box_y * top.box_z;
    const double prefac = COULOMB_KCAL_A / (PI_VAL * vol);
    const double pi2_over_a2 = (PI_VAL * PI_VAL) / (top.ewald_alpha * top.ewald_alpha);

    double u_recip = 0.0;
    ws.mesh[0] = {0.0, 0.0}; // Conducting (tin-foil) boundary + neutralizing background plasma

    for (int mx = 0; mx < kx; ++mx) {
        const int mpx = (mx <= kx / 2) ? mx : (mx - kx);
        const double rx_m = static_cast<double>(mpx) / top.box_x;
        const double bx = ws.mod_x[mx];

        for (int my = 0; my < ky; ++my) {
            const int mpy = (my <= ky / 2) ? my : (my - ky);
            const double ry_m = static_cast<double>(mpy) / top.box_y;
            const double bxy = bx * ws.mod_y[my];

            for (int mz = 0; mz < kz; ++mz) {
                if (mx == 0 && my == 0 && mz == 0) continue;
                const int mpz = (mz <= kz / 2) ? mz : (mz - kz);
                const double rz_m = static_cast<double>(mpz) / top.box_z;

                const double m2 = rx_m * rx_m + ry_m * ry_m + rz_m * rz_m;
                const double G = prefac * (std::exp(-pi2_over_a2 * m2) / m2) * (bxy * ws.mod_z[mz]);

                const size_t idx = static_cast<size_t>(mx * ky + my) * kz + mz;
                const std::complex<double> qh = ws.mesh[idx];
                u_recip += 0.5 * G * std::norm(qh);
                ws.mesh[idx] = G * qh;
            }
        }
    }
    eb.elec_recip += u_recip;

    // 4. Unnormalized Inverse 3D FFT: G(m)*Q_hat(m) -> Phi(k) = dU_recip / dQ(k)
    detail::fft_3d(ws.mesh, kx, ky, kz, -1);

    // 5. Analytical B-spline differentiation for exact conservative forces F_i = -grad U_recip
    for (int32_t i = 0; i < n; ++i) {
        const double qi = top.charge[i];
        if (qi == 0.0) continue;

        double fx = rx[i] / top.box_x; fx -= std::floor(fx);
        double fy = ry[i] / top.box_y; fy -= std::floor(fy);
        double fz = rz[i] / top.box_z; fz -= std::floor(fz);

        const double ux = fx * kx;
        const double uy = fy * ky;
        const double uz = fz * kz;

        const int gx0 = static_cast<int>(std::floor(ux)) % kx;
        const int gy0 = static_cast<int>(std::floor(uy)) % ky;
        const int gz0 = static_cast<int>(std::floor(uz)) % kz;

        double wx[4], dwx[4], wy[4], dwy[4], wz[4], dwz[4];
        detail::bspline4_weights(ux - std::floor(ux), wx, dwx);
        detail::bspline4_weights(uy - std::floor(uy), wy, dwy);
        detail::bspline4_weights(uz - std::floor(uz), wz, dwz);

        double dU_dux = 0.0, dU_duy = 0.0, dU_duz = 0.0;
        for (int px = 0; px < 4; ++px) {
            const int ix = (gx0 - px + kx) % kx;
            for (int py = 0; py < 4; ++py) {
                const int iy = (gy0 - py + ky) % ky;
                const size_t base = static_cast<size_t>(ix * ky + iy) * kz;
                for (int pz = 0; pz < 4; ++pz) {
                    const int iz = (gz0 - pz + kz) % kz;
                    const double phi = ws.mesh[base + iz].real();
                    dU_dux += phi * dwx[px] *  wy[py] *  wz[pz];
                    dU_duy += phi *  wx[px] * dwy[py] *  wz[pz];
                    dU_duz += phi *  wx[px] *  wy[py] * dwz[pz];
                }
            }
        }

        qfx[i] += double_to_qforce(-qi * scale_ux * dU_dux);
        qfy[i] += double_to_qforce(-qi * scale_uy * dU_duy);
        qfz[i] += double_to_qforce(-qi * scale_uz * dU_duz);
    }
}

} // namespace ashmd
