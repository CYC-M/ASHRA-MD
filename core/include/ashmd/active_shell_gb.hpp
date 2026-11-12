#pragma once

// Original Pillar 2: Active-Shell Coupled Hamiltonian (ASH Mode)
//   - OBC II (Onufriev-Bashford-Case 2004, igb=5: alpha=1.0, beta=0.8, gamma=4.85)
//     Generalized Born solvation coupling BOTH solute AND explicit interfacial shell waters!
//   - Physiological Debye-Huckel salt screening kappa(I, T) in Still's polarization kernel
//   - Complete analytical Born-radius chain-rule forces (dU_GB/d_alpha_i + dU_NP/d_alpha_i) * d_alpha_i/dr_ij
//   - ACE-type smooth nonpolar hydrophobic cavity hydration energy: 4*pi*gamma*(rho_i + R_probe)^2*(rho_i/alpha_i)^6
//   - Globally C^infinity Softmin Pocket-Envelope Hydration Cap with exact Newton's 3rd Law reaction forces
//   - Three-zone topological partition (skipping redundant C-C scaffold internal pairs)

#include "types.hpp"
#include "parallel.hpp"
#include <cmath>
#include <vector>
#include <algorithm>

namespace ashmd {

class ASHWorkspace {
public:
    std::vector<NeighborPair> nb_pairs;      // Non-excluded short/mid-range pairs (!both_scaf)
    std::vector<NeighborPair> inter_pairs;   // Interface non-excluded pairs for O(N_inter) IHRT cross-potentials
    std::vector<std::pair<int32_t, int32_t>> gb_all_pairs;     // Active-involving pairs (!both_scaf) within gb_cutoff+skin
    std::vector<NeighborPair> gb_charged_pairs;                // Subset of gb_all_pairs with q_i * q_j != 0
    std::vector<int64_t> qI_scaf_static;  // Exact constant descreening integral between frozen scaffold-scaffold pairs
    std::vector<double> rho_tilde;        // Precomputed max(0.8, gb_radius[i] - 0.09)
    std::vector<double> scaled_r;         // Precomputed gb_screen[i] * rho_tilde[i]
    std::vector<int32_t> mobile_indices;
    std::vector<double> last_x, last_y, last_z;
    double skin = 2.0;
    bool initialized = false;
    bool scaf_static_ready = false;

    std::vector<double> psi;         // Descreening sum Psi_i * rho_tilde_i
    std::vector<double> born_r;      // Effective Born radius alpha_i [A]
    std::vector<double> dalpha_dI;   // d(alpha_i) / d(I_sum_i) where I_sum_i = sum_j I_{ij}
    std::vector<double> dU_dalpha;   // Total partial derivative d(U_GB + U_NP)/d(alpha_i)
    std::vector<int64_t> qI_sum;
    std::vector<double> I_sum;
    std::vector<qenergy_t> qdU_da_pair;
    std::vector<double> B_chain;
    std::vector<double> cap_s_dist;
    std::vector<double> cap_exp_val;

    std::vector<std::vector<int64_t>> tls_fx, tls_fy, tls_fz;
    std::vector<std::vector<int64_t>> tls_vec_acc;
    std::vector<std::vector<qenergy_t>> tls_du_da;
    std::vector<qenergy_t> tls_e1, tls_e2, tls_e3;

    void ensure_buffers(int32_t n_threads, int32_t n_atoms) {
        psi.assign(n_atoms, 0.0);
        born_r.assign(n_atoms, 1.5);
        dalpha_dI.assign(n_atoms, 0.0);
        dU_dalpha.assign(n_atoms, 0.0);
        if (static_cast<int32_t>(I_sum.size()) != n_atoms) {
            I_sum.resize(n_atoms, 0.0);
            qdU_da_pair.resize(n_atoms, 0);
            B_chain.resize(n_atoms, 0.0);
        }

        if (static_cast<int32_t>(tls_fx.size()) != n_threads) {
            tls_fx.resize(n_threads);
            tls_fy.resize(n_threads);
            tls_fz.resize(n_threads);
            tls_vec_acc.resize(n_threads);
            tls_du_da.resize(n_threads);
            tls_e1.resize(n_threads);
            tls_e2.resize(n_threads);
            tls_e3.resize(n_threads);
        }
        for (int32_t t = 0; t < n_threads; ++t) {
            tls_fx[t].assign(n_atoms, 0);
            tls_fy[t].assign(n_atoms, 0);
            tls_fz[t].assign(n_atoms, 0);
            tls_vec_acc[t].assign(n_atoms, 0);
            tls_du_da[t].assign(n_atoms, 0);
            tls_e1[t] = 0;
            tls_e2[t] = 0;
            tls_e3[t] = 0;
        }
    }

    bool needs_rebuild(const SystemTopology& top, const double* rx, const double* ry, const double* rz) const noexcept {
        if (!initialized || static_cast<int32_t>(last_x.size()) != top.n_atoms) return true;
        const double thresh_sq = 0.25 * skin * skin;
        for (int32_t i : mobile_indices) {
            const double dx = rx[i] - last_x[i];
            const double dy = ry[i] - last_y[i];
            const double dz = rz[i] - last_z[i];
            if (dx * dx + dy * dy + dz * dz > thresh_sq) return true;
        }
        return false;
    }

    void rebuild_lists(const SystemTopology& top, const double* rx, const double* ry, const double* rz);
};

namespace detail {

// HCT / OBC2 analytical descreening integral I(r; rho_tilde_i, s_j) and exact dI/dr
// With smooth C2 cutoff feathering between [gb_sw, gb_cut] so Born radii are C2 smooth everywhere!
inline void obc_pair_descreen(
    double r,
    double rho_ti,
    double sj,
    double gb_sw,
    double gb_cut,
    double& I_val,
    double& dI_dr
) noexcept {
    I_val = 0.0;
    dI_dr = 0.0;
    if (r >= gb_cut || r + sj <= rho_ti || sj <= 0.0) return;

    // Standard HCT regime (avoid interior complement singularity if extreme clash r + rho_ti <= sj)
    if (r + rho_ti <= sj) return;

    const double L = std::max(rho_ti, r - sj);
    const double U = r + sj;
    const double invL = 1.0 / L;
    const double invU = 1.0 / U;
    const double invL2 = invL * invL;
    const double invU2 = invU * invU;
    const double inv_r = 1.0 / std::max(r, 1.0e-12);
    const double logLU = std::log(L * invU);
    const double sj2   = sj * sj;
    const double r2    = r * r;

    const double raw_I = 0.5 * (
        (invL - invU)
        + 0.25 * (sj2 - r2) * inv_r * (invL2 - invU2)
        + 0.5 * inv_r * logLU
    );
    // Exact Leibniz derivative (boundary terms vanish identically in both L=rho_ti and L=r-sj!)
    const double raw_dI = 0.5 * (
        -0.25 * (sj2 + r2) * (inv_r * inv_r) * (invL2 - invU2)
        - 0.5 * (inv_r * inv_r) * logLU
    );

    if (r > gb_sw && gb_cut > gb_sw) {
        const double inv_w = 1.0 / (gb_cut - gb_sw);
        const double u  = (r - gb_sw) * inv_w;
        const double u2 = u * u;
        const double u3 = u2 * u;
        const double sw  = 1.0 - 10.0 * u3 + 15.0 * u2 * u2 - 6.0 * u3 * u2;
        const double dsw = (-30.0 * u2 + 60.0 * u3 - 30.0 * u2 * u2) * inv_w;
        I_val = sw * raw_I;
        dI_dr = sw * raw_dI + dsw * raw_I;
    } else {
        I_val = raw_I;
        dI_dr = raw_dI;
    }
}

} // namespace detail

inline void ASHWorkspace::rebuild_lists(
    const SystemTopology& top,
    const double* rx,
    const double* ry,
    const double* rz
) {
    const int32_t n = top.n_atoms;
    last_x.assign(rx, rx + n);
    last_y.assign(ry, ry + n);
    last_z.assign(rz, rz + n);

    mobile_indices.clear();
    std::vector<int32_t> scaf_indices;
    mobile_indices.reserve(n);
    scaf_indices.reserve(n);
    for (int32_t i = 0; i < n; ++i) {
        if (top.zone[i] == ZONE_SCAFFOLD) scaf_indices.push_back(i);
        else mobile_indices.push_back(i);
    }

    if (static_cast<int32_t>(rho_tilde.size()) != n) {
        rho_tilde.resize(n);
        scaled_r.resize(n);
        for (int32_t i = 0; i < n; ++i) {
            rho_tilde[i] = std::max(0.8, top.gb_radius[i] - 0.09);
            scaled_r[i]  = top.gb_screen[i] * rho_tilde[i];
        }
    }

    if (!scaf_static_ready || static_cast<int32_t>(qI_scaf_static.size()) != n) {
        qI_scaf_static.assign(n, 0);
        const double gb_cut = top.gb_cutoff;
        const double gb_sw  = std::max(0.0, gb_cut - 2.0);
        const double gb_cut_sq = gb_cut * gb_cut;
        const int32_t n_sc = static_cast<int32_t>(scaf_indices.size());
        for (int32_t a = 0; a < n_sc; ++a) {
            const int32_t i = scaf_indices[a];
            const double rho_ti = rho_tilde[i];
            const double si     = scaled_r[i];
            for (int32_t b = a + 1; b < n_sc; ++b) {
                const int32_t j = scaf_indices[b];
                const double dx = rx[i] - rx[j];
                const double dy = ry[i] - ry[j];
                const double dz = rz[i] - rz[j];
                const double r2 = dx * dx + dy * dy + dz * dz;
                if (r2 >= gb_cut_sq) continue;
                const double r = std::sqrt(std::max(r2, 1.0e-18));
                const double rho_tj = rho_tilde[j];
                const double sj     = scaled_r[j];
                double I_ij = 0.0, dI_ij = 0.0, I_ji = 0.0, dI_ji = 0.0;
                detail::obc_pair_descreen(r, rho_ti, sj, gb_sw, gb_cut, I_ij, dI_ij);
                detail::obc_pair_descreen(r, rho_tj, si, gb_sw, gb_cut, I_ji, dI_ji);
                qI_scaf_static[i] += double_to_qdescreen(I_ij);
                qI_scaf_static[j] += double_to_qdescreen(I_ji);
            }
        }
        scaf_static_ready = true;
    }

    const double rlist_sq = (top.gb_cutoff + skin) * (top.gb_cutoff + skin);
    nb_pairs.clear();
    inter_pairs.clear();
    gb_all_pairs.clear();
    gb_charged_pairs.clear();
    nb_pairs.reserve(mobile_indices.size() * 96);
    inter_pairs.reserve(1024);
    gb_all_pairs.reserve(mobile_indices.size() * 108);
    gb_charged_pairs.reserve(mobile_indices.size() * 104);

    // Visit every !both_scaf pair (i < j) in exact lexicographical order without touching O(N_scaf^2)
    auto mob_it = mobile_indices.begin();
    for (int32_t i = 0; i < n; ++i) {
        while (mob_it != mobile_indices.end() && *mob_it <= i) ++mob_it;
        const bool qi_nz = (top.charge[i] != 0.0);
        if (top.zone[i] != ZONE_SCAFFOLD) {
            for (int32_t j = i + 1; j < n; ++j) {
                const double dx = rx[i] - rx[j];
                const double dy = ry[i] - ry[j];
                const double dz = rz[i] - rz[j];
                if (dx * dx + dy * dy + dz * dz <= rlist_sq) {
                    const bool inter = (top.role[i] == ROLE_PEPTIDE && (top.role[j] == ROLE_RECEPTOR || top.role[j] == ROLE_ION)) ||
                                       (top.role[j] == ROLE_PEPTIDE && (top.role[i] == ROLE_RECEPTOR || top.role[i] == ROLE_ION));
                    gb_all_pairs.push_back({i, j});
                    if (qi_nz && top.charge[j] != 0.0) {
                        gb_charged_pairs.push_back({i, j, inter});
                    }
                    if (!top.is_excluded(i, j)) {
                        nb_pairs.push_back({i, j, inter});
                        if (inter) inter_pairs.push_back({i, j, true});
                    }
                }
            }
        } else {
            for (auto it = mob_it; it != mobile_indices.end(); ++it) {
                const int32_t j = *it;
                const double dx = rx[i] - rx[j];
                const double dy = ry[i] - ry[j];
                const double dz = rz[i] - rz[j];
                if (dx * dx + dy * dy + dz * dz <= rlist_sq) {
                    const bool inter = (top.role[i] == ROLE_PEPTIDE && (top.role[j] == ROLE_RECEPTOR || top.role[j] == ROLE_ION)) ||
                                       (top.role[j] == ROLE_PEPTIDE && (top.role[i] == ROLE_RECEPTOR || top.role[i] == ROLE_ION));
                    gb_all_pairs.push_back({i, j});
                    if (qi_nz && top.charge[j] != 0.0) {
                        gb_charged_pairs.push_back({i, j, inter});
                    }
                    if (!top.is_excluded(i, j)) {
                        nb_pairs.push_back({i, j, inter});
                        if (inter) inter_pairs.push_back({i, j, true});
                    }
                }
            }
        }
    }
    initialized = true;
}

// Short/Mid-Range Vacuum Electrostatics + Switched LJ + Softmin Hydration Cap for ASH Mode
inline void compute_ash_direct_and_cap(
    const SystemTopology& top,
    ASHWorkspace& ws,
    ParallelPool& pool,
    const double* __restrict rx,
    const double* __restrict ry,
    const double* __restrict rz,
    const ReplicaTemperingParams& rep,
    int64_t* __restrict qfx,
    int64_t* __restrict qfy,
    int64_t* __restrict qfz,
    EnergyBreakdown& eb,
    double target_temp_k = PHYSIOLOGICAL_TEMP_K
) {
    (void)target_temp_k;
    if (ws.needs_rebuild(top, rx, ry, rz)) {
        ws.rebuild_lists(top, rx, ry, rz);
    }

    const int32_t n = top.n_atoms;
    const int32_t n_threads = pool.num_threads();
    ws.ensure_buffers(n_threads, n);

    const double rcut = top.gb_cutoff;
    const double rcut_sq = rcut * rcut;
    const double ron = std::max(0.0, rcut - 2.0);
    const double inv_sw = 1.0 / std::max(rcut - ron, 1.0e-6);

    const int32_t npairs = static_cast<int32_t>(ws.nb_pairs.size());
    pool.parallel_for(npairs, 256, [&](int32_t tid, int32_t beg, int32_t end) {
        auto* lfx = ws.tls_fx[tid].data();
        auto* lfy = ws.tls_fy[tid].data();
        auto* lfz = ws.tls_fz[tid].data();
        qenergy_t qe_lj_loc = 0;
        qenergy_t qe_el_loc = 0;
        qenergy_t qe_raw_inter = 0;

        for (int32_t idx = beg; idx < end; ++idx) {
            const auto& pr = ws.nb_pairs[idx];
            const int32_t i = pr.i;
            const int32_t j = pr.j;

            const double dx = rx[i] - rx[j];
            const double dy = ry[i] - ry[j];
            const double dz = rz[i] - rz[j];
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 >= rcut_sq) continue;

            const double r = std::sqrt(std::max(r2, 1.0e-18));
            const double inv_r = 1.0 / r;
            const double scale = pr.is_interface ? rep.lambda_inter : 1.0;

            // C2 switched Coulomb & LJ so long-time trajectories conserve energy cleanly
            double sw = 1.0, dsw = 0.0;
            if (r > ron) {
                const double u = (r - ron) * inv_sw;
                const double u2 = u * u;
                const double u3 = u2 * u;
                sw  = 1.0 - 10.0 * u3 + 15.0 * u2 * u2 - 6.0 * u3 * u2;
                dsw = (-30.0 * u2 + 60.0 * u3 - 30.0 * u2 * u2) * inv_sw;
            }

            const double sc_a = pr.is_interface ? rep.softcore_alpha : 0.0;

            // 1. Direct Coulomb (1 / eps_in) with Steinbrecher-Joung-Case soft-core (when sc_a > 0) & C2 switch
            const double qq = (COULOMB_KCAL_A / top.eps_in) * top.charge[i] * top.charge[j];
            double u_el_bare = 0.0, du_el_bare = 0.0;
            if (sc_a > 0.0) {
                const double r2_el   = r2 + 12.0 * sc_a;
                const double inv_rel = 1.0 / std::sqrt(r2_el);
                u_el_bare  = qq * inv_rel;
                du_el_bare = -qq * r * inv_rel * inv_rel * inv_rel;
            } else {
                u_el_bare  = qq * inv_r;
                du_el_bare = -qq * inv_r * inv_r;
            }
            const double u_el_unsc  = sw * u_el_bare;
            const double du_el_unsc = sw * du_el_bare + dsw * u_el_bare;

            // 2. 12-6 LJ (with soft-core if interface tempered)
            const double eps_ij  = std::sqrt(top.epsilon[i] * top.epsilon[j]);
            const double rmin_ij = top.rmin_half[i] + top.rmin_half[j];
            double u_lj_unsc = 0.0, du_lj_unsc = 0.0;
            if (eps_ij > 0.0) {
                double u_lj_bare = 0.0, du_lj_bare = 0.0;
                if (sc_a > 0.0) {
                    const double r6  = r2 * r2 * r2;
                    const double rm2 = rmin_ij * rmin_ij;
                    const double rm6 = rm2 * rm2 * rm2;
                    const double den = r6 + sc_a * rm6;
                    const double x   = rm6 / den;
                    u_lj_bare  = eps_ij * (x * x - 2.0 * x);
                    du_lj_bare = eps_ij * (2.0 * x - 2.0) * (-6.0 * x * (r2 * r2 * r) / den);
                } else {
                    const double sr2 = (rmin_ij * rmin_ij) / r2;
                    const double sr6 = sr2 * sr2 * sr2;
                    u_lj_bare  = eps_ij * (sr6 * sr6 - 2.0 * sr6);
                    du_lj_bare = -12.0 * eps_ij * (sr6 * sr6 - sr6) * inv_r;
                }
                u_lj_unsc  = sw * u_lj_bare;
                du_lj_unsc = sw * du_lj_bare + dsw * u_lj_bare;
            }

            if (pr.is_interface) qe_raw_inter += double_to_qenergy(u_el_unsc + u_lj_unsc);
            qe_el_loc += double_to_qenergy(scale * u_el_unsc);
            qe_lj_loc += double_to_qenergy(scale * u_lj_unsc);

            const double f_over_r = -scale * (du_el_unsc + du_lj_unsc) * inv_r;
            const int64_t fx = double_to_qforce(f_over_r * dx);
            const int64_t fy = double_to_qforce(f_over_r * dy);
            const int64_t fz = double_to_qforce(f_over_r * dz);
            lfx[i] += fx; lfy[i] += fy; lfz[i] += fz;
            lfx[j] -= fx; lfy[j] -= fy; lfz[j] -= fz;
        }
        ws.tls_e1[tid] += qe_lj_loc;
        ws.tls_e2[tid] += qe_el_loc;
        ws.tls_e3[tid] += qe_raw_inter;
    });

    qenergy_t qe_lj_sum = 0, qe_el_sum = 0, qe_inter_sum = 0;
    for (int32_t t = 0; t < n_threads; ++t) {
        qe_lj_sum += ws.tls_e1[t];
        qe_el_sum += ws.tls_e2[t];
        qe_inter_sum += ws.tls_e3[t];
        for (int32_t i = 0; i < n; ++i) {
            qfx[i] += ws.tls_fx[t][i];
            qfy[i] += ws.tls_fy[t][i];
            qfz[i] += ws.tls_fz[t][i];
        }
    }
    eb.lj_sr += qenergy_to_double(qe_lj_sum);
    eb.elec_dir += qenergy_to_double(qe_el_sum);
    eb.raw_interface_nonb += qenergy_to_double(qe_inter_sum);

    // 3. C^infinity Softmin Pocket-Envelope Hydration Cap (exact Newton's 3rd law!)
    if (!top.shell_water_oxygens.empty() && !top.cap_anchors.empty() && top.cap_k > 0.0) {
        const double beta = top.cap_soft_beta;
        const double rcap = top.cap_radius;
        const double kcap = top.cap_k;
        constexpr double EPS_SMOOTH_SQ = 0.01; // 0.1 A smoothing core -> globally C^infinity

        const int32_t n_anc = static_cast<int32_t>(top.cap_anchors.size());
        if (static_cast<int32_t>(ws.cap_s_dist.size()) < n_anc) {
            ws.cap_s_dist.resize(n_anc);
            ws.cap_exp_val.resize(n_anc);
        }
        auto* s_dist  = ws.cap_s_dist.data();
        auto* exp_val = ws.cap_exp_val.data();

        double e_cap = 0.0;
        for (int32_t w : top.shell_water_oxygens) {
            double min_s = 1.0e30;
            for (int32_t a = 0; a < n_anc; ++a) {
                const int32_t p = top.cap_anchors[a];
                const double dx = rx[w] - rx[p];
                const double dy = ry[w] - ry[p];
                const double dz = rz[w] - rz[p];
                const double s  = std::sqrt(dx * dx + dy * dy + dz * dz + EPS_SMOOTH_SQ);
                s_dist[a] = s;
                if (s < min_s) min_s = s;
            }
            // Log-sum-exp shift by min_s prevents overflow/underflow completely
            double Z = 0.0;
            for (int32_t a = 0; a < n_anc; ++a) {
                const double ev = std::exp(-beta * (s_dist[a] - min_s));
                exp_val[a] = ev;
                Z += ev;
            }
            const double d_env = min_s - std::log(Z) / beta;
            const double u = d_env - rcap;
            if (u > 0.0) {
                e_cap += 0.5 * kcap * u * u;
                const double dU_dd = kcap * u;
                const double inv_Z = 1.0 / Z;

                int64_t sum_fw_x = 0, sum_fw_y = 0, sum_fw_z = 0;
                for (int32_t a = 0; a < n_anc; ++a) {
                    const int32_t p = top.cap_anchors[a];
                    const double weight = exp_val[a] * inv_Z;
                    const double dx = rx[w] - rx[p];
                    const double dy = ry[w] - ry[p];
                    const double dz = rz[w] - rz[p];
                    const double coef = -dU_dd * weight / s_dist[a];
                    // Force contribution on w is coef * (r_w - r_p); equal & opposite on p!
                    const int64_t fwx = double_to_qforce(coef * dx);
                    const int64_t fwy = double_to_qforce(coef * dy);
                    const int64_t fwz = double_to_qforce(coef * dz);
                    sum_fw_x += fwx; sum_fw_y += fwy; sum_fw_z += fwz;
                    qfx[p] -= fwx;   qfy[p] -= fwy;   qfz[p] -= fwz;
                }
                qfx[w] += sum_fw_x; qfy[w] += sum_fw_y; qfz[w] += sum_fw_z;
            }
        }
        eb.cap_shell += e_cap;
    }
}

// Fast exact O(N_inter) evaluator for ASH direct Coulomb + LJ interface energy under arbitrary (lambda, softcore_alpha)
// Eliminates redundant O(N^2) + Born-radius evaluations during IHRT swap / MBAR cross-energy checks.
inline double eval_ash_interface_direct_lj_only(
    const SystemTopology& top,
    ASHWorkspace& ws,
    const double* __restrict rx,
    const double* __restrict ry,
    const double* __restrict rz,
    double lambda_inter,
    double softcore_alpha
) {
    if (ws.needs_rebuild(top, rx, ry, rz)) {
        ws.rebuild_lists(top, rx, ry, rz);
    }
    if (lambda_inter == 0.0 || ws.inter_pairs.empty()) return 0.0;

    const double rcut = top.gb_cutoff;
    const double rcut_sq = rcut * rcut;
    const double ron = std::max(0.0, rcut - 2.0);
    const double inv_sw = 1.0 / std::max(rcut - ron, 1.0e-6);
    const double c_over_eps = COULOMB_KCAL_A / top.eps_in;

    qenergy_t qe_lj = 0, qe_el = 0;
    for (const auto& pr : ws.inter_pairs) {
        const int32_t i = pr.i;
        const int32_t j = pr.j;
        const double dx = rx[i] - rx[j];
        const double dy = ry[i] - ry[j];
        const double dz = rz[i] - rz[j];
        const double r2 = dx * dx + dy * dy + dz * dz;
        if (r2 >= rcut_sq) continue;

        const double r = std::sqrt(std::max(r2, 1.0e-18));
        const double inv_r = 1.0 / r;
        double sw = 1.0;
        if (r > ron) {
            const double u = (r - ron) * inv_sw;
            const double u2 = u * u;
            const double u3 = u2 * u;
            sw = 1.0 - 10.0 * u3 + 15.0 * u2 * u2 - 6.0 * u3 * u2;
        }

        const double qq = c_over_eps * top.charge[i] * top.charge[j];
        const double u_el_bare = (softcore_alpha > 0.0)
            ? (qq / std::sqrt(r2 + 12.0 * softcore_alpha))
            : (qq * inv_r);
        const double u_el_unsc = sw * u_el_bare;

        const double eps_ij  = std::sqrt(top.epsilon[i] * top.epsilon[j]);
        const double rmin_ij = top.rmin_half[i] + top.rmin_half[j];
        double u_lj_unsc = 0.0;
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
            u_lj_unsc = sw * u_lj_bare;
        }

        qe_el += double_to_qenergy(lambda_inter * u_el_unsc);
        qe_lj += double_to_qenergy(lambda_inter * u_lj_unsc);
    }
    return qenergy_to_double(qe_lj) + qenergy_to_double(qe_el);
}

// OBC II Generalized Born Polarization + ACE Hydrophobic Nonpolar Solvation
// Can be evaluated every step OR every slow FC-BAOAB outer step on filtered coordinates!
inline void compute_ash_gb_and_nonpolar(
    const SystemTopology& top,
    ASHWorkspace& ws,
    ParallelPool& pool,
    const double* __restrict rx,
    const double* __restrict ry,
    const double* __restrict rz,
    const ReplicaTemperingParams& rep,
    int64_t* __restrict qfx,
    int64_t* __restrict qfy,
    int64_t* __restrict qfz,
    EnergyBreakdown& eb,
    double target_temp_k = PHYSIOLOGICAL_TEMP_K
) {
    if (ws.needs_rebuild(top, rx, ry, rz)) {
        ws.rebuild_lists(top, rx, ry, rz);
    }

    const int32_t n = top.n_atoms;
    const int32_t n_threads = pool.num_threads();
    ws.ensure_buffers(n_threads, n);

    const double gb_cut = top.gb_cutoff;
    const double gb_sw  = std::max(0.0, gb_cut - 2.0);
    const double gb_cut_sq = gb_cut * gb_cut;
    const double* __restrict rho_tilde_arr = ws.rho_tilde.data();
    const double* __restrict scaled_r_arr  = ws.scaled_r.data();

    // Step 1: Accumulate pairwise descreening integral I_sum_i = sum_{j != i} I_{ij}
    const int32_t n_gb_pairs = static_cast<int32_t>(ws.gb_all_pairs.size());
    pool.parallel_for(n_gb_pairs, 256, [&](int32_t tid, int32_t beg, int32_t end) {
        auto* acc = ws.tls_vec_acc[tid].data();
        for (int32_t idx = beg; idx < end; ++idx) {
            const int32_t i = ws.gb_all_pairs[idx].first;
            const int32_t j = ws.gb_all_pairs[idx].second;
            const double dx = rx[i] - rx[j];
            const double dy = ry[i] - ry[j];
            const double dz = rz[i] - rz[j];
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 >= gb_cut_sq) continue;
            const double r = std::sqrt(std::max(r2, 1.0e-18));

            double I_ij = 0.0, dI_ij = 0.0;
            double I_ji = 0.0, dI_ji = 0.0;
            detail::obc_pair_descreen(r, rho_tilde_arr[i], scaled_r_arr[j], gb_sw, gb_cut, I_ij, dI_ij);
            detail::obc_pair_descreen(r, rho_tilde_arr[j], scaled_r_arr[i], gb_sw, gb_cut, I_ji, dI_ji);

            acc[i] += double_to_qdescreen(I_ij);
            acc[j] += double_to_qdescreen(I_ji);
        }
    });

    ws.qI_sum = ws.qI_scaf_static;
    if (static_cast<int32_t>(ws.qI_sum.size()) != n) ws.qI_sum.assign(n, 0);
    for (int32_t t = 0; t < n_threads; ++t) {
        for (int32_t i = 0; i < n; ++i) ws.qI_sum[i] += ws.tls_vec_acc[t][i];
        std::fill(ws.tls_vec_acc[t].begin(), ws.tls_vec_acc[t].end(), 0);
    }
    for (int32_t i = 0; i < n; ++i) {
        ws.I_sum[i] = qdescreen_to_double(ws.qI_sum[i]);
    }

    // Step 2: OBC II Born radii alpha_i, d(alpha_i)/d(I_sum_i), self GB energy, and ACE nonpolar energy
    // OBC II parameters: a=1.0, b=0.8, c=4.85
    constexpr double OBC_A = 1.0;
    constexpr double OBC_B = 0.8;
    constexpr double OBC_C = 4.85;
    constexpr double PROBE_R = 1.4; // Water probe radius [A]

    double kappa = 0.0;
    if (top.ionic_strength > 0.0) {
        const double num = 8.0 * PI_VAL * COULOMB_KCAL_A * 6.02214076e-4 * top.ionic_strength;
        const double den = top.eps_out * BOLTZMANN_KCAL * std::max(target_temp_k, 1.0);
        kappa = std::sqrt(num / den);
    }
    const double inv_eps_in  = 1.0 / top.eps_in;
    const double inv_eps_out = 1.0 / top.eps_out;

    double e_gb_self = 0.0;
    double e_np      = 0.0;
    for (int32_t i = 0; i < n; ++i) {
        const double rho_i  = std::max(0.89, top.gb_radius[i]);
        const double rho_ti = rho_i - 0.09;
        const double psi_i  = rho_ti * ws.I_sum[i];
        const double psi2   = psi_i * psi_i;
        const double psi3   = psi2 * psi_i;
        const double arg    = OBC_A * psi_i - OBC_B * psi2 + OBC_C * psi3;
        const double th     = std::tanh(arg);
        const double sech2  = 1.0 - th * th;

        const double inv_alpha = (1.0 / rho_ti) - th / rho_i;
        const double alpha_i   = 1.0 / std::max(inv_alpha, 1.0e-4);
        ws.born_r[i] = alpha_i;

        const double darg_dpsi = OBC_A - 2.0 * OBC_B * psi_i + 3.0 * OBC_C * psi2;
        ws.dalpha_dI[i] = (alpha_i * alpha_i) * (rho_ti / rho_i) * sech2 * darg_dpsi;

        const double qi = top.charge[i];
        if (qi != 0.0) {
            const double exp_ka = std::exp(-kappa * alpha_i);
            const double diel   = inv_eps_in - exp_ka * inv_eps_out;
            const double c_self = -0.5 * COULOMB_KCAL_A * qi * qi;
            e_gb_self += c_self * diel / alpha_i;
            ws.dU_dalpha[i] += c_self * ((kappa * exp_ka * inv_eps_out) / alpha_i - diel / (alpha_i * alpha_i));
        }

        if (top.sasa_gamma > 0.0 && top.role[i] != ROLE_SOLVENT) {
            const double rp   = rho_i + PROBE_R;
            const double rat  = rho_i / alpha_i;
            const double rat6 = (rat * rat * rat) * (rat * rat * rat);
            const double u_np_i = 4.0 * PI_VAL * top.sasa_gamma * (rp * rp) * rat6;
            e_np += u_np_i;
            ws.dU_dalpha[i] += -6.0 * u_np_i / alpha_i;
        }
    }

    // Step 3: Pairwise GB polarization energy, direct radial forces, and dU/dalpha_i accumulation
    const int32_t n_gb_chg = static_cast<int32_t>(ws.gb_charged_pairs.size());
    pool.parallel_for(n_gb_chg, 256, [&](int32_t tid, int32_t beg, int32_t end) {
        auto* lfx = ws.tls_fx[tid].data();
        auto* lfy = ws.tls_fy[tid].data();
        auto* lfz = ws.tls_fz[tid].data();
        auto* ldU_da = ws.tls_du_da[tid].data();
        qenergy_t qe_gb_pair = 0;
        qenergy_t qe_raw_inter = 0;

        for (int32_t idx = beg; idx < end; ++idx) {
            const auto& pr = ws.gb_charged_pairs[idx];
            const int32_t i = pr.i;
            const int32_t j = pr.j;

            const double dx = rx[i] - rx[j];
            const double dy = ry[i] - ry[j];
            const double dz = rz[i] - rz[j];
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 >= gb_cut_sq) continue;
            const double r = std::sqrt(std::max(r2, 1.0e-18));

            double sw = 1.0, dsw = 0.0;
            if (r > gb_sw && gb_cut > gb_sw) {
                const double inv_w = 1.0 / (gb_cut - gb_sw);
                const double u  = (r - gb_sw) * inv_w;
                const double u2 = u * u;
                const double u3 = u2 * u;
                sw  = 1.0 - 10.0 * u3 + 15.0 * u2 * u2 - 6.0 * u3 * u2;
                dsw = (-30.0 * u2 + 60.0 * u3 - 30.0 * u2 * u2) * inv_w;
            }

            const double qq = COULOMB_KCAL_A * top.charge[i] * top.charge[j];
            const double scale = pr.is_interface ? rep.lambda_inter : 1.0;

            const double ai = ws.born_r[i];
            const double aj = ws.born_r[j];
            const double aij = ai * aj;
            const double D   = r2 / (4.0 * aij);
            const double exp_mD = std::exp(-D);
            const double f2     = r2 + aij * exp_mD;
            const double f_gb   = std::sqrt(f2);
            const double inv_f  = 1.0 / f_gb;

            const double exp_kf = std::exp(-kappa * f_gb);
            const double diel   = inv_eps_in - exp_kf * inv_eps_out;

            const double u_bare = -qq * diel * inv_f;
            const double u_sw   = sw * u_bare;

            if (pr.is_interface) qe_raw_inter += double_to_qenergy(u_sw);
            qe_gb_pair += double_to_qenergy(scale * u_sw);

            const double du_bare_df = -qq * ((kappa * exp_kf * inv_eps_out) * inv_f - diel * inv_f * inv_f);
            const double df_dr = (r * inv_f) * (1.0 - 0.25 * exp_mD);
            const double du_dr_total = scale * (sw * du_bare_df * df_dr + dsw * u_bare);

            const double f_over_r = -du_dr_total / r;
            const int64_t fx = double_to_qforce(f_over_r * dx);
            const int64_t fy = double_to_qforce(f_over_r * dy);
            const int64_t fz = double_to_qforce(f_over_r * dz);
            lfx[i] += fx; lfy[i] += fy; lfz[i] += fz;
            lfx[j] -= fx; lfy[j] -= fy; lfz[j] -= fz;

            const double df_daij = 0.5 * inv_f * exp_mD * (1.0 + D);
            const double common_da = scale * sw * du_bare_df * df_daij;
            ldU_da[i] += double_to_qenergy(common_da * aj);
            ldU_da[j] += double_to_qenergy(common_da * ai);
        }
        ws.tls_e1[tid] += qe_gb_pair;
        ws.tls_e2[tid] += qe_raw_inter;
    });

    qenergy_t qe_gb_pair_sum = 0;
    qenergy_t qe_raw_inter_sum = 0;
    std::fill(ws.qdU_da_pair.begin(), ws.qdU_da_pair.end(), 0);
    for (int32_t t = 0; t < n_threads; ++t) {
        qe_gb_pair_sum += ws.tls_e1[t];
        qe_raw_inter_sum += ws.tls_e2[t];
        for (int32_t i = 0; i < n; ++i) {
            ws.qdU_da_pair[i] += ws.tls_du_da[t][i];
            qfx[i] += ws.tls_fx[t][i];
            qfy[i] += ws.tls_fy[t][i];
            qfz[i] += ws.tls_fz[t][i];
            ws.tls_fx[t][i] = 0;
            ws.tls_fy[t][i] = 0;
            ws.tls_fz[t][i] = 0;
        }
    }
    for (int32_t i = 0; i < n; ++i) {
        ws.dU_dalpha[i] += qenergy_to_double(ws.qdU_da_pair[i]);
        ws.B_chain[i]    = ws.dU_dalpha[i] * ws.dalpha_dI[i];
    }
    eb.gb_pol  += e_gb_self + qenergy_to_double(qe_gb_pair_sum);
    eb.raw_interface_nonb += qenergy_to_double(qe_raw_inter_sum);
    eb.np_sasa += e_np;

    // Step 4: Chain-rule Born-radius forces
    const double* __restrict B_chain_arr = ws.B_chain.data();
    pool.parallel_for(n_gb_pairs, 256, [&](int32_t tid, int32_t beg, int32_t end) {
        auto* lfx = ws.tls_fx[tid].data();
        auto* lfy = ws.tls_fy[tid].data();
        auto* lfz = ws.tls_fz[tid].data();

        for (int32_t idx = beg; idx < end; ++idx) {
            const int32_t i = ws.gb_all_pairs[idx].first;
            const int32_t j = ws.gb_all_pairs[idx].second;
            const double dx = rx[i] - rx[j];
            const double dy = ry[i] - ry[j];
            const double dz = rz[i] - rz[j];
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 >= gb_cut_sq) continue;
            const double r = std::sqrt(std::max(r2, 1.0e-18));

            double I_ij = 0.0, dI_ij = 0.0;
            double I_ji = 0.0, dI_ji = 0.0;
            detail::obc_pair_descreen(r, rho_tilde_arr[i], scaled_r_arr[j], gb_sw, gb_cut, I_ij, dI_ij);
            detail::obc_pair_descreen(r, rho_tilde_arr[j], scaled_r_arr[i], gb_sw, gb_cut, I_ji, dI_ji);

            const double dU_dr_born = B_chain_arr[i] * dI_ij + B_chain_arr[j] * dI_ji;
            if (dU_dr_born == 0.0) continue;

            const double f_over_r = -dU_dr_born / r;
            const int64_t fx = double_to_qforce(f_over_r * dx);
            const int64_t fy = double_to_qforce(f_over_r * dy);
            const int64_t fz = double_to_qforce(f_over_r * dz);
            lfx[i] += fx; lfy[i] += fy; lfz[i] += fz;
            lfx[j] -= fx; lfy[j] -= fy; lfz[j] -= fz;
        }
    });

    for (int32_t t = 0; t < n_threads; ++t) {
        for (int32_t i = 0; i < n; ++i) {
            qfx[i] += ws.tls_fx[t][i];
            qfy[i] += ws.tls_fy[t][i];
            qfz[i] += ws.tls_fz[t][i];
        }
    }
}

// Exact Infinite-Cutoff MM/OBC2-GBSA Binding Free Energy Decomposition on a Trajectory Frame:
//   Delta_G_MMGBSA = [U_AB - U_A - U_B] = dE_vdW + dE_elec + dG_GB(OBC2 + Debye-Huckel) + dG_NP(ACE)
// Eliminates finite-cutoff truncation on charged complexes and captures exact Born-radius desolvation penalties.
inline void compute_exact_mmgbsa_binding(
    const SystemTopology& top,
    ParallelPool& pool,
    const double* __restrict rx,
    const double* __restrict ry,
    const double* __restrict rz,
    double target_temp_k,
    double sasa_gamma_gbsa,
    double eps_in_gbsa,
    double out_terms[5]
) {
    for (int k = 0; k < 5; ++k) out_terms[k] = 0.0;
    std::vector<int32_t> grp_A, grp_B, grp_AB;
    grp_A.reserve(top.n_atoms);
    grp_B.reserve(top.n_atoms);
    grp_AB.reserve(top.n_atoms);
    for (int32_t i = 0; i < top.n_atoms; ++i) {
        if (top.role[i] == ROLE_RECEPTOR || top.role[i] == ROLE_ION) {
            grp_A.push_back(i);
            grp_AB.push_back(i);
        } else if (top.role[i] == ROLE_PEPTIDE) {
            grp_B.push_back(i);
            grp_AB.push_back(i);
        }
    }
    if (grp_A.empty() || grp_B.empty()) return;

    double kappa = 0.0;
    if (top.ionic_strength > 0.0) {
        const double num = 8.0 * PI_VAL * COULOMB_KCAL_A * 6.02214076e-4 * top.ionic_strength;
        const double den = top.eps_out * BOLTZMANN_KCAL * std::max(target_temp_k, 1.0);
        kappa = std::sqrt(num / den);
    }
    const double eff_eps_in  = (eps_in_gbsa > 0.0) ? eps_in_gbsa : top.eps_in;
    const double inv_eps_in  = 1.0 / eff_eps_in;
    const double inv_eps_out = 1.0 / top.eps_out;
    constexpr double OBC_A = 1.0, OBC_B = 0.8, OBC_C = 4.85, PROBE_R = 1.4;
    const int32_t nt = pool.num_threads();

    auto eval_subset_solv = [&](const std::vector<int32_t>& sub, double& gb_out, double& np_out) {
        const int32_t m = static_cast<int32_t>(sub.size());
        std::vector<double> I_sub(m, 0.0);
        pool.parallel_for(m, 16, [&](int32_t /*tid*/, int32_t beg, int32_t end) {
            for (int32_t a = beg; a < end; ++a) {
                const int32_t i = sub[a];
                const double rho_ti = std::max(0.8, top.gb_radius[i] - 0.09);
                double acc = 0.0;
                for (int32_t b = 0; b < m; ++b) {
                    if (a == b) continue;
                    const int32_t j = sub[b];
                    const double dx = rx[i] - rx[j];
                    const double dy = ry[i] - ry[j];
                    const double dz = rz[i] - rz[j];
                    const double r  = std::sqrt(std::max(dx*dx + dy*dy + dz*dz, 1.0e-18));
                    const double rho_tj = std::max(0.8, top.gb_radius[j] - 0.09);
                    const double sj = top.gb_screen[j] * rho_tj;
                    double I_ij = 0.0, dI_ij = 0.0;
                    detail::obc_pair_descreen(r, rho_ti, sj, 1.0e9, 1.0e9, I_ij, dI_ij);
                    acc += I_ij;
                }
                I_sub[a] = acc;
            }
        });

        std::vector<double> alpha(m, 1.5);
        qenergy_t qe_gb_self = 0;
        qenergy_t qe_np = 0;
        for (int32_t a = 0; a < m; ++a) {
            const int32_t i = sub[a];
            const double rho_i  = std::max(0.89, top.gb_radius[i]);
            const double rho_ti = rho_i - 0.09;
            const double psi    = rho_ti * I_sub[a];
            const double arg    = OBC_A * psi - OBC_B * psi * psi + OBC_C * psi * psi * psi;
            const double al     = 1.0 / std::max((1.0 / rho_ti) - std::tanh(arg) / rho_i, 1.0e-4);
            alpha[a] = al;

            const double qi = top.charge[i];
            if (qi != 0.0) {
                const double diel = inv_eps_in - std::exp(-kappa * al) * inv_eps_out;
                qe_gb_self += double_to_qenergy(-0.5 * COULOMB_KCAL_A * qi * qi * diel / al);
            }
            if (sasa_gamma_gbsa > 0.0) {
                const double rp = rho_i + PROBE_R;
                const double rat = rho_i / al;
                const double rat6 = (rat * rat * rat) * (rat * rat * rat);
                qe_np += double_to_qenergy(4.0 * PI_VAL * sasa_gamma_gbsa * rp * rp * rat6);
            }
        }

        std::vector<qenergy_t> tls_gb_pair(nt, 0);
        pool.parallel_for(m, 16, [&](int32_t tid, int32_t beg, int32_t end) {
            qenergy_t loc = 0;
            for (int32_t a = beg; a < end; ++a) {
                const int32_t i = sub[a];
                const double qi = top.charge[i];
                if (qi == 0.0) continue;
                const double ai = alpha[a];
                for (int32_t b = a + 1; b < m; ++b) {
                    const int32_t j = sub[b];
                    const double qq = COULOMB_KCAL_A * qi * top.charge[j];
                    if (qq == 0.0) continue;
                    const double dx = rx[i] - rx[j];
                    const double dy = ry[i] - ry[j];
                    const double dz = rz[i] - rz[j];
                    const double r2 = dx*dx + dy*dy + dz*dz;
                    const double aij = ai * alpha[b];
                    const double f_gb = std::sqrt(r2 + aij * std::exp(-r2 / (4.0 * aij)));
                    const double diel = inv_eps_in - std::exp(-kappa * f_gb) * inv_eps_out;
                    loc += double_to_qenergy(-qq * diel / f_gb);
                }
            }
            tls_gb_pair[tid] += loc;
        });

        qenergy_t qe_pair_tot = 0;
        for (int32_t t = 0; t < nt; ++t) qe_pair_tot += tls_gb_pair[t];
        gb_out = qenergy_to_double(qe_gb_self + qe_pair_tot);
        np_out = qenergy_to_double(qe_np);
    };

    double gb_AB = 0.0, np_AB = 0.0, gb_A = 0.0, np_A = 0.0, gb_B = 0.0, np_B = 0.0;
    eval_subset_solv(grp_AB, gb_AB, np_AB);
    eval_subset_solv(grp_A,  gb_A,  np_A);
    eval_subset_solv(grp_B,  gb_B,  np_B);

    const int32_t mA = static_cast<int32_t>(grp_A.size());
    const int32_t mB = static_cast<int32_t>(grp_B.size());
    std::vector<qenergy_t> tls_vdw(nt, 0), tls_el(nt, 0);
    pool.parallel_for(mA, 16, [&](int32_t tid, int32_t beg, int32_t end) {
        qenergy_t l_vdw = 0, l_el = 0;
        for (int32_t a = beg; a < end; ++a) {
            const int32_t i = grp_A[a];
            const double qi = top.charge[i];
            const double epsi = top.epsilon[i];
            const double rmi  = top.rmin_half[i];
            for (int32_t b = 0; b < mB; ++b) {
                const int32_t j = grp_B[b];
                const double dx = rx[i] - rx[j];
                const double dy = ry[i] - ry[j];
                const double dz = rz[i] - rz[j];
                const double r2 = std::max(dx*dx + dy*dy + dz*dz, 1.0e-12);
                const double inv_r = 1.0 / std::sqrt(r2);
                if (qi != 0.0 && top.charge[j] != 0.0) {
                    l_el += double_to_qenergy((COULOMB_KCAL_A * inv_eps_in) * qi * top.charge[j] * inv_r);
                }
                const double eps_ij = std::sqrt(epsi * top.epsilon[j]);
                if (eps_ij > 0.0) {
                    const double rm = rmi + top.rmin_half[j];
                    const double sr2 = (rm * rm) / r2;
                    const double sr6 = sr2 * sr2 * sr2;
                    l_vdw += double_to_qenergy(eps_ij * (sr6 * sr6 - 2.0 * sr6));
                }
            }
        }
        tls_vdw[tid] += l_vdw;
        tls_el[tid]  += l_el;
    });

    qenergy_t q_vdw = 0, q_el = 0;
    for (int32_t t = 0; t < nt; ++t) {
        q_vdw += tls_vdw[t];
        q_el  += tls_el[t];
    }
    out_terms[0] = qenergy_to_double(q_vdw);
    out_terms[1] = qenergy_to_double(q_el);
    out_terms[2] = gb_AB - gb_A - gb_B;
    out_terms[3] = np_AB - np_A - np_B;
    out_terms[4] = out_terms[0] + out_terms[1] + out_terms[2] + out_terms[3];
}

// Deterministic 92-point Golden-Spiral Fibonacci Shrake-Rupley Connolly SASA [A^2]
// Uses spatial 6.6 A sphere-overlap filtering + last-burier caching (< 0.2 ms for 2,000 atoms).
inline double compute_connolly_sasa_subset(
    const SystemTopology& top,
    ParallelPool& pool,
    const std::vector<int32_t>& sub,
    const double* __restrict rx,
    const double* __restrict ry,
    const double* __restrict rz,
    std::vector<double>& sasa_sub_out,
    double probe_r = 1.40
) {
    constexpr int32_t N_PTS = 92;
    struct UnitVec3 { double x, y, z; };
    static const std::array<UnitVec3, N_PTS> fib_pts = []() {
        std::array<UnitVec3, N_PTS> pts{};
        const double golden_angle = PI_VAL * (3.0 - std::sqrt(5.0));
        for (int32_t p = 0; p < N_PTS; ++p) {
            const double z = 1.0 - (2.0 * (static_cast<double>(p) + 0.5)) / static_cast<double>(N_PTS);
            const double rad = std::sqrt(std::max(0.0, 1.0 - z * z));
            const double phi = golden_angle * static_cast<double>(p);
            pts[p] = {rad * std::cos(phi), rad * std::sin(phi), z};
        }
        return pts;
    }();

    const int32_t m = static_cast<int32_t>(sub.size());
    sasa_sub_out.assign(m, 0.0);
    if (m == 0) return 0.0;

    std::vector<double> rad_exp(m), rad_exp_sq(m);
    for (int32_t a = 0; a < m; ++a) {
        const int32_t i = sub[a];
        const double r_vdw = (top.rmin_half[i] > 0.1) ? std::max(1.0, top.gb_radius[i]) : 0.0;
        rad_exp[a] = (r_vdw > 0.0) ? (r_vdw + probe_r) : 0.0;
        rad_exp_sq[a] = rad_exp[a] * rad_exp[a];
    }

    pool.parallel_for(m, 16, [&](int32_t /*tid*/, int32_t beg, int32_t end) {
        std::vector<int32_t> nbrs;
        nbrs.reserve(64);
        for (int32_t a = beg; a < end; ++a) {
            const double Ri = rad_exp[a];
            if (Ri <= 0.0) continue;
            const int32_t i = sub[a];
            const double xi = rx[i], yi = ry[i], zi = rz[i];

            nbrs.clear();
            for (int32_t b = 0; b < m; ++b) {
                if (a == b || rad_exp[b] <= 0.0) continue;
                const int32_t j = sub[b];
                const double dx = xi - rx[j];
                const double dy = yi - ry[j];
                const double dz = zi - rz[j];
                const double rsum = Ri + rad_exp[b];
                const double d2 = dx * dx + dy * dy + dz * dz;
                if (d2 < rsum * rsum) {
                    nbrs.push_back(b);
                }
            }

            if (nbrs.empty()) {
                sasa_sub_out[a] = 4.0 * PI_VAL * rad_exp_sq[a];
                continue;
            }

            int32_t exposed = 0;
            int32_t last_burier = nbrs[0];
            const int32_t n_nb = static_cast<int32_t>(nbrs.size());

            for (int32_t p = 0; p < N_PTS; ++p) {
                const double px = xi + Ri * fib_pts[p].x;
                const double py = yi + Ri * fib_pts[p].y;
                const double pz = zi + Ri * fib_pts[p].z;

                // O(1) check against most recent burying sphere
                {
                    const int32_t j_lb = sub[last_burier];
                    const double dx = px - rx[j_lb];
                    const double dy = py - ry[j_lb];
                    const double dz = pz - rz[j_lb];
                    if (dx * dx + dy * dy + dz * dz < rad_exp_sq[last_burier]) {
                        continue;
                    }
                }

                bool buried = false;
                for (int32_t k = 0; k < n_nb; ++k) {
                    const int32_t b = nbrs[k];
                    if (b == last_burier) continue;
                    const int32_t j = sub[b];
                    const double dx = px - rx[j];
                    const double dy = py - ry[j];
                    const double dz = pz - rz[j];
                    if (dx * dx + dy * dy + dz * dz < rad_exp_sq[b]) {
                        last_burier = b;
                        buried = true;
                        break;
                    }
                }
                if (!buried) ++exposed;
            }

            sasa_sub_out[a] = (4.0 * PI_VAL * rad_exp_sq[a]) * (static_cast<double>(exposed) / static_cast<double>(N_PTS));
        }
    });

    qenergy_t q_sasa = 0;
    for (int32_t a = 0; a < m; ++a) q_sasa += double_to_qenergy(sasa_sub_out[a]);
    return qenergy_to_double(q_sasa);
}

// High-Accuracy First-Principles Physical Binding Free Energy Decomposition (v2.0):
//   1) Rank-1 Positive-Definite Residue-Dielectric OBC2 + Debye-Huckel (RD-OBC2; Hou et al. JCIM 2011):
//        eps_ij = sqrt(eps_i * eps_j), preserving exact Debye-Huckel water asymptotics as r -> inf
//   2) Exact 92-point Connolly SASA Hydrophobic Cavity Hydration (Wagoner & Baker PNAS 2006):
//        dG_cav = cav_gamma * (SASA_AB - SASA_A - SASA_B) < 0 upon burial
//   3) Exact Analytical WCA Continuum TIP3P Water Dispersion Compensation (Levy, Zhang, Gallicchio & Felts JACS 2003;
//      Gallicchio & Levy JCC 2004):
//        dG_disp_solv = G_disp(AB) - G_disp(A) - G_disp(B) > 0 upon burial (solute-water vdW lost when water displaced)
//   4) Exact per-atom unbound (sasa_free_out) & bound (sasa_bound_out) Connolly SASA [A^2] for burial-weighted entropy.
inline void compute_rd_gbsa_wca_binding(
    const SystemTopology& top,
    ParallelPool& pool,
    const double* __restrict rx,
    const double* __restrict ry,
    const double* __restrict rz,
    double target_temp_k,
    double cav_gamma,
    double disp_scale,
    double out_terms[8],
    double* sasa_free_out = nullptr,
    double* sasa_bound_out = nullptr
) {
    for (int k = 0; k < 8; ++k) out_terms[k] = 0.0;
    if (sasa_free_out)  std::fill(sasa_free_out,  sasa_free_out + top.n_atoms, 0.0);
    if (sasa_bound_out) std::fill(sasa_bound_out, sasa_bound_out + top.n_atoms, 0.0);

    std::vector<int32_t> grp_A, grp_B, grp_AB;
    grp_A.reserve(top.n_atoms);
    grp_B.reserve(top.n_atoms);
    grp_AB.reserve(top.n_atoms);
    for (int32_t i = 0; i < top.n_atoms; ++i) {
        if (top.role[i] == ROLE_RECEPTOR || top.role[i] == ROLE_ION) {
            grp_A.push_back(i);
            grp_AB.push_back(i);
        } else if (top.role[i] == ROLE_PEPTIDE) {
            grp_B.push_back(i);
            grp_AB.push_back(i);
        }
    }
    if (grp_A.empty() || grp_B.empty()) return;

    double kappa = 0.0;
    if (top.ionic_strength > 0.0) {
        const double num = 8.0 * PI_VAL * COULOMB_KCAL_A * 6.02214076e-4 * top.ionic_strength;
        const double den = top.eps_out * BOLTZMANN_KCAL * std::max(target_temp_k, 1.0);
        kappa = std::sqrt(num / den);
    }
    const double inv_eps_out = 1.0 / top.eps_out;

    std::vector<double> inv_sqrt_eps(top.n_atoms, 1.0 / std::sqrt(2.0));
    for (int32_t i = 0; i < top.n_atoms; ++i) {
        const double ei = (i < static_cast<int32_t>(top.res_diel.size()) && top.res_diel[i] > 0.0)
                        ? top.res_diel[i]
                        : 2.0;
        inv_sqrt_eps[i] = 1.0 / std::sqrt(ei);
    }

    // Physical TIP3P continuum liquid water dispersion constants at 1.0 g/cm^3 (zero fitted parameters):
    //   rho_w = 0.033428 molecules / A^3, eps_w = 0.1520 kcal/mol, rmin_half_w = 1.7683 A
    //   Exact WCA attractive integral prefactor: 4 * pi * [5/9 + (1 - 2^{-1/2})/3] * rho_w = 0.274384688 A^{-3}
    constexpr double RHO_WATER_A3    = 0.033428;
    constexpr double EPS_WATER_TIP3P = 0.1520;
    constexpr double RMIN_HALF_WATER = 1.7683;
    const double wca_prefac = 4.0 * PI_VAL * ((5.0 / 9.0) + (1.0 - std::pow(2.0, -0.5)) / 3.0) * RHO_WATER_A3 * disp_scale;

    constexpr double OBC_A = 1.0, OBC_B = 0.8, OBC_C = 4.85;
    const int32_t nt = pool.num_threads();

    auto eval_subset_rd_solv = [&](
        const std::vector<int32_t>& sub,
        double& gb_rd_out,
        double& disp_out,
        double& sasa_tot_out,
        std::vector<double>& sasa_vec_out
    ) {
        const int32_t m = static_cast<int32_t>(sub.size());
        std::vector<double> I_sub(m, 0.0);
        pool.parallel_for(m, 16, [&](int32_t /*tid*/, int32_t beg, int32_t end) {
            for (int32_t a = beg; a < end; ++a) {
                const int32_t i = sub[a];
                const double rho_ti = std::max(0.8, top.gb_radius[i] - 0.09);
                double acc = 0.0;
                for (int32_t b = 0; b < m; ++b) {
                    if (a == b) continue;
                    const int32_t j = sub[b];
                    const double dx = rx[i] - rx[j];
                    const double dy = ry[i] - ry[j];
                    const double dz = rz[i] - rz[j];
                    const double r  = std::sqrt(std::max(dx*dx + dy*dy + dz*dz, 1.0e-18));
                    const double rho_tj = std::max(0.8, top.gb_radius[j] - 0.09);
                    const double sj = top.gb_screen[j] * rho_tj;
                    double I_ij = 0.0, dI_ij = 0.0;
                    detail::obc_pair_descreen(r, rho_ti, sj, 1.0e9, 1.0e9, I_ij, dI_ij);
                    acc += I_ij;
                }
                I_sub[a] = acc;
            }
        });

        std::vector<double> alpha(m, 1.5);
        qenergy_t qe_gb_self = 0;
        qenergy_t qe_disp = 0;
        for (int32_t a = 0; a < m; ++a) {
            const int32_t i = sub[a];
            const double rho_i  = std::max(0.89, top.gb_radius[i]);
            const double rho_ti = rho_i - 0.09;
            const double psi    = rho_ti * I_sub[a];
            const double arg    = OBC_A * psi - OBC_B * psi * psi + OBC_C * psi * psi * psi;
            const double al     = 1.0 / std::max((1.0 / rho_ti) - std::tanh(arg) / rho_i, 1.0e-4);
            alpha[a] = al;

            const double qi = top.charge[i];
            if (qi != 0.0) {
                const double inv_ei = inv_sqrt_eps[i] * inv_sqrt_eps[i];
                const double diel   = inv_ei - std::exp(-kappa * al) * inv_eps_out;
                qe_gb_self += double_to_qenergy(-0.5 * COULOMB_KCAL_A * qi * qi * diel / al);
            }

            // Published AGBNP rule (Gallicchio & Levy, J. Comput. Chem. 2004):
            // Hydrogens participate in volume descreening I_sub, while WCA continuum water
            // dispersion spheres are integrated over solute heavy atoms (mass > 2.0 amu) using OBC2 al.
            if (disp_scale > 0.0 && top.mass[i] > 2.0 && top.epsilon[i] > 0.0 && top.rmin_half[i] > 0.1) {
                const double R_iw = top.rmin_half[i] + RMIN_HALF_WATER;
                const double B_i  = std::max(R_iw, al + (R_iw - rho_i));
                const double ratio = R_iw / B_i;
                const double ratio3 = ratio * ratio * ratio;
                const double R_iw3  = R_iw * R_iw * R_iw;
                const double g_disp_i = -wca_prefac * std::sqrt(top.epsilon[i] * EPS_WATER_TIP3P) * R_iw3 * ratio3;
                qe_disp += double_to_qenergy(g_disp_i);
            }
        }

        std::vector<qenergy_t> tls_gb_pair(nt, 0);
        pool.parallel_for(m, 16, [&](int32_t tid, int32_t beg, int32_t end) {
            qenergy_t loc = 0;
            for (int32_t a = beg; a < end; ++a) {
                const int32_t i = sub[a];
                const double qi = top.charge[i];
                if (qi == 0.0) continue;
                const double ai = alpha[a];
                const double ise_i = inv_sqrt_eps[i];
                for (int32_t b = a + 1; b < m; ++b) {
                    const int32_t j = sub[b];
                    const double qq = COULOMB_KCAL_A * qi * top.charge[j];
                    if (qq == 0.0) continue;
                    const double dx = rx[i] - rx[j];
                    const double dy = ry[i] - ry[j];
                    const double dz = rz[i] - rz[j];
                    const double r2 = dx*dx + dy*dy + dz*dz;
                    const double aij = ai * alpha[b];
                    const double f_gb = std::sqrt(r2 + aij * std::exp(-r2 / (4.0 * aij)));
                    const double inv_eps_ij = ise_i * inv_sqrt_eps[j];
                    const double diel = inv_eps_ij - std::exp(-kappa * f_gb) * inv_eps_out;
                    loc += double_to_qenergy(-qq * diel / f_gb);
                }
            }
            tls_gb_pair[tid] += loc;
        });

        qenergy_t qe_pair_tot = 0;
        for (int32_t t = 0; t < nt; ++t) qe_pair_tot += tls_gb_pair[t];
        gb_rd_out    = qenergy_to_double(qe_gb_self + qe_pair_tot);
        disp_out     = qenergy_to_double(qe_disp);
        sasa_tot_out = compute_connolly_sasa_subset(top, pool, sub, rx, ry, rz, sasa_vec_out);
    };

    double gb_AB = 0.0, disp_AB = 0.0, sasa_AB = 0.0;
    double gb_A  = 0.0, disp_A  = 0.0, sasa_A  = 0.0;
    double gb_B  = 0.0, disp_B  = 0.0, sasa_B  = 0.0;
    std::vector<double> sv_AB, sv_A, sv_B;

    eval_subset_rd_solv(grp_AB, gb_AB, disp_AB, sasa_AB, sv_AB);
    eval_subset_rd_solv(grp_A,  gb_A,  disp_A,  sasa_A,  sv_A);
    eval_subset_rd_solv(grp_B,  gb_B,  disp_B,  sasa_B,  sv_B);

    if (sasa_bound_out) {
        for (size_t k = 0; k < grp_AB.size(); ++k) sasa_bound_out[grp_AB[k]] = sv_AB[k];
    }
    if (sasa_free_out) {
        for (size_t k = 0; k < grp_A.size(); ++k) sasa_free_out[grp_A[k]] = sv_A[k];
        for (size_t k = 0; k < grp_B.size(); ++k) sasa_free_out[grp_B[k]] = sv_B[k];
    }

    const int32_t mA = static_cast<int32_t>(grp_A.size());
    const int32_t mB = static_cast<int32_t>(grp_B.size());
    std::vector<qenergy_t> tls_vdw(nt, 0), tls_el(nt, 0);
    pool.parallel_for(mA, 16, [&](int32_t tid, int32_t beg, int32_t end) {
        qenergy_t l_vdw = 0, l_el = 0;
        for (int32_t a = beg; a < end; ++a) {
            const int32_t i = grp_A[a];
            const double qi = top.charge[i];
            const double ise_i = inv_sqrt_eps[i];
            const double epsi = top.epsilon[i];
            const double rmi  = top.rmin_half[i];
            for (int32_t b = 0; b < mB; ++b) {
                const int32_t j = grp_B[b];
                const double dx = rx[i] - rx[j];
                const double dy = ry[i] - ry[j];
                const double dz = rz[i] - rz[j];
                const double r2 = std::max(dx*dx + dy*dy + dz*dz, 1.0e-12);
                const double inv_r = 1.0 / std::sqrt(r2);
                if (qi != 0.0 && top.charge[j] != 0.0) {
                    const double inv_eps_ij = ise_i * inv_sqrt_eps[j];
                    l_el += double_to_qenergy((COULOMB_KCAL_A * inv_eps_ij) * qi * top.charge[j] * inv_r);
                }
                const double eps_ij = std::sqrt(epsi * top.epsilon[j]);
                if (eps_ij > 0.0) {
                    const double rm = rmi + top.rmin_half[j];
                    const double sr2 = (rm * rm) / r2;
                    const double sr6 = sr2 * sr2 * sr2;
                    l_vdw += double_to_qenergy(eps_ij * (sr6 * sr6 - 2.0 * sr6));
                }
            }
        }
        tls_vdw[tid] += l_vdw;
        tls_el[tid]  += l_el;
    });

    qenergy_t q_vdw = 0, q_el = 0;
    for (int32_t t = 0; t < nt; ++t) {
        q_vdw += tls_vdw[t];
        q_el  += tls_el[t];
    }

    const double d_sasa = sasa_AB - sasa_A - sasa_B; // Negative upon burial
    out_terms[0] = qenergy_to_double(q_vdw);         // dE_vdw
    out_terms[1] = disp_AB - disp_A - disp_B;        // dG_disp_solv (> 0 upon burial)
    out_terms[2] = cav_gamma * d_sasa;               // dG_cav_sasa  (< 0 upon burial)
    out_terms[3] = qenergy_to_double(q_el);          // dE_elec_rd
    out_terms[4] = gb_AB - gb_A - gb_B;              // dG_gb_rd
    out_terms[5] = std::max(0.0, -d_sasa);           // buried_sasa_a2 [A^2]
    out_terms[6] = out_terms[0] + out_terms[1] + out_terms[2]; // net nonpolar
    out_terms[7] = out_terms[6] + out_terms[3] + out_terms[4]; // total enthalpy + solvation
}

} // namespace ashmd
