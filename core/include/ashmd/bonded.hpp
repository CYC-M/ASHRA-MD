#pragma once

// Exact analytical bonded potentials & forces (AMBER99SB-ILDN conventions):
//   - Harmonic bonds: U = kr * (r - req)^2
//   - Harmonic angles: U = ktheta * (theta - theta_eq)^2 (singularity-free atan2 cross-product gradient)
//   - Periodic dihedrals (proper & improper): U = pk * (1 + cos(n * phi - phase))
//     with Blondel & Karplus (1996) singularity-free zero-torque Cartesian derivatives
//   - 1-4 scaled AMBER nonbonded pairs (SCEE=1.2, SCNB=2.0)
//   - Harmonic Zone-B buffer / equilibration tethers

#include "types.hpp"
#include <cmath>
#include <vector>

namespace ashmd {

inline void compute_bonded_and_tether(
    const SystemTopology& top,
    const double* __restrict rx,
    const double* __restrict ry,
    const double* __restrict rz,
    const ReplicaTemperingParams& rep,
    int64_t* __restrict qfx,
    int64_t* __restrict qfy,
    int64_t* __restrict qfz,
    EnergyBreakdown& eb,
    bool include_14 = true
) noexcept {
    double e_bond = 0.0;
    for (const auto& b : top.bonds) {
        const double dx = rx[b.i] - rx[b.j];
        const double dy = ry[b.i] - ry[b.j];
        const double dz = rz[b.i] - rz[b.j];
        const double r2 = dx * dx + dy * dy + dz * dz;
        const double r  = std::sqrt(std::max(r2, 1.0e-24));
        const double dr = r - b.req;
        e_bond += b.kr * dr * dr;

        const double f_over_r = -2.0 * b.kr * dr / r;
        const int64_t fx = double_to_qforce(f_over_r * dx);
        const int64_t fy = double_to_qforce(f_over_r * dy);
        const int64_t fz = double_to_qforce(f_over_r * dz);

        qfx[b.i] += fx; qfy[b.i] += fy; qfz[b.i] += fz;
        qfx[b.j] -= fx; qfy[b.j] -= fy; qfz[b.j] -= fz;
    }
    eb.bond += e_bond;

    double e_angle = 0.0;
    for (const auto& a : top.angles) {
        const double ux = rx[a.i] - rx[a.j];
        const double uy = ry[a.i] - ry[a.j];
        const double uz = rz[a.i] - rz[a.j];

        const double vx = rx[a.k] - rx[a.j];
        const double vy = ry[a.k] - ry[a.j];
        const double vz = rz[a.k] - rz[a.j];

        const double ru2 = std::max(ux * ux + uy * uy + uz * uz, 1.0e-24);
        const double rv2 = std::max(vx * vx + vy * vy + vz * vz, 1.0e-24);

        const double wx = uy * vz - uz * vy;
        const double wy = uz * vx - ux * vz;
        const double wz = ux * vy - uy * vx;
        const double w  = std::sqrt(wx * wx + wy * wy + wz * wz);
        const double dot_uv = ux * vx + uy * vy + uz * vz;

        const double theta = std::atan2(w, dot_uv);
        const double dth   = theta - a.theta_eq;
        e_angle += a.ktheta * dth * dth;

        if (w > 1.0e-12) {
            const double dU_dth = 2.0 * a.ktheta * dth;
            const double inv_ru2_w = 1.0 / (ru2 * w);
            const double inv_rv2_w = 1.0 / (rv2 * w);

            // u x w
            const double uxw_x = uy * wz - uz * wy;
            const double uxw_y = uz * wx - ux * wz;
            const double uxw_z = ux * wy - uy * wx;

            // w x v
            const double wxv_x = wy * vz - wz * vy;
            const double wxv_y = wz * vx - wx * vz;
            const double wxv_z = wx * vy - wy * vx;

            const int64_t fix = double_to_qforce(-dU_dth * uxw_x * inv_ru2_w);
            const int64_t fiy = double_to_qforce(-dU_dth * uxw_y * inv_ru2_w);
            const int64_t fiz = double_to_qforce(-dU_dth * uxw_z * inv_ru2_w);

            const int64_t fkx = double_to_qforce(-dU_dth * wxv_x * inv_rv2_w);
            const int64_t fky = double_to_qforce(-dU_dth * wxv_y * inv_rv2_w);
            const int64_t fkz = double_to_qforce(-dU_dth * wxv_z * inv_rv2_w);

            qfx[a.i] += fix; qfy[a.i] += fiy; qfz[a.i] += fiz;
            qfx[a.k] += fkx; qfy[a.k] += fky; qfz[a.k] += fkz;
            qfx[a.j] -= (fix + fkx);
            qfy[a.j] -= (fiy + fky);
            qfz[a.j] -= (fiz + fkz);
        }
    }
    eb.angle += e_angle;

    double e_dih = 0.0;
    double raw_pep_dih = 0.0;
    for (const auto& d : top.dihedrals) {
        const double b1x = rx[d.j] - rx[d.i];
        const double b1y = ry[d.j] - ry[d.i];
        const double b1z = rz[d.j] - rz[d.i];

        const double b2x = rx[d.k] - rx[d.j];
        const double b2y = ry[d.k] - ry[d.j];
        const double b2z = rz[d.k] - rz[d.j];

        const double b3x = rx[d.l] - rx[d.k];
        const double b3y = ry[d.l] - ry[d.k];
        const double b3z = rz[d.l] - rz[d.k];

        const double mx = b1y * b2z - b1z * b2y;
        const double my = b1z * b2x - b1x * b2z;
        const double mz = b1x * b2y - b1y * b2x;

        const double nx = b2y * b3z - b2z * b3y;
        const double ny = b2z * b3x - b2x * b3z;
        const double nz = b2x * b3y - b2y * b3x;

        const double m2  = mx * mx + my * my + mz * mz;
        const double n2  = nx * nx + ny * ny + nz * nz;
        const double b22 = b2x * b2x + b2y * b2y + b2z * b2z;

        if (m2 < 1.0e-20 || n2 < 1.0e-20 || b22 < 1.0e-20) continue;

        const double b2_norm = std::sqrt(b22);
        const double y_arg   = b2_norm * (b1x * nx + b1y * ny + b1z * nz);
        const double x_arg   = mx * nx + my * ny + mz * nz;
        const double phi     = std::atan2(y_arg, x_arg);

        const double scale = d.is_peptide_torsion ? rep.mu_torsion : 1.0;
        const double arg   = d.periodicity * phi - d.phase;
        const double term_raw = d.pk * (1.0 + std::cos(arg));
        if (d.is_peptide_torsion) raw_pep_dih += term_raw;
        e_dih += scale * term_raw;

        const double dU_dphi = -scale * d.pk * d.periodicity * std::sin(arg);

        const double gi_fac = -b2_norm / m2;
        const double gl_fac =  b2_norm / n2;

        const double gix = gi_fac * mx;
        const double giy = gi_fac * my;
        const double giz = gi_fac * mz;

        const double glx = gl_fac * nx;
        const double gly = gl_fac * ny;
        const double glz = gl_fac * nz;

        const double inv_b22 = 1.0 / b22;
        const double c12 = (b1x * b2x + b1y * b2y + b1z * b2z) * inv_b22;
        const double c32 = (b3x * b2x + b3y * b2y + b3z * b2z) * inv_b22;

        const double sx = c12 * gix - c32 * glx;
        const double sy = c12 * giy - c32 * gly;
        const double sz = c12 * giz - c32 * glz;

        const int64_t fix = double_to_qforce(-dU_dphi * gix);
        const int64_t fiy = double_to_qforce(-dU_dphi * giy);
        const int64_t fiz = double_to_qforce(-dU_dphi * giz);

        const int64_t flx = double_to_qforce(-dU_dphi * glx);
        const int64_t fly = double_to_qforce(-dU_dphi * gly);
        const int64_t flz = double_to_qforce(-dU_dphi * glz);

        const int64_t fjx = double_to_qforce(-dU_dphi * (-gix - sx));
        const int64_t fjy = double_to_qforce(-dU_dphi * (-giy - sy));
        const int64_t fjz = double_to_qforce(-dU_dphi * (-giz - sz));

        // Exact force sum conservation on k: fk = -(fi + fj + fl)
        const int64_t fkx = -(fix + fjx + flx);
        const int64_t fky = -(fiy + fjy + fly);
        const int64_t fkz = -(fiz + fjz + flz);

        qfx[d.i] += fix; qfy[d.i] += fiy; qfz[d.i] += fiz;
        qfx[d.j] += fjx; qfy[d.j] += fjy; qfz[d.j] += fjz;
        qfx[d.k] += fkx; qfy[d.k] += fky; qfz[d.k] += fkz;
        qfx[d.l] += flx; qfy[d.l] += fly; qfz[d.l] += flz;
    }
    eb.dihedral += e_dih;
    eb.raw_pep_torsion += raw_pep_dih;

    if (include_14) {
        double e_14_lj = 0.0;
        double e_14_el = 0.0;
        for (const auto& p : top.pairs14) {
            const double dx = rx[p.i] - rx[p.j];
            const double dy = ry[p.i] - ry[p.j];
            const double dz = rz[p.i] - rz[p.j];
            const double r2 = std::max(dx * dx + dy * dy + dz * dz, 1.0e-12);
            const double inv_r  = 1.0 / std::sqrt(r2);
            const double inv_r2 = inv_r * inv_r;

            const double el = p.q_prod_scaled * inv_r;
            e_14_el += el;

            double sr2 = (p.rmin_ij * p.rmin_ij) * inv_r2;
            double sr6 = sr2 * sr2 * sr2;
            double lj  = p.eps_scaled * (sr6 * sr6 - 2.0 * sr6);
            e_14_lj += lj;

            // -dU/dr / r = (el + 12 * eps * (sr12 - sr6)) / r^2
            const double f_over_r = (el + 12.0 * p.eps_scaled * (sr6 * sr6 - sr6)) * inv_r2;
            const int64_t fx = double_to_qforce(f_over_r * dx);
            const int64_t fy = double_to_qforce(f_over_r * dy);
            const int64_t fz = double_to_qforce(f_over_r * dz);

            qfx[p.i] += fx; qfy[p.i] += fy; qfz[p.i] += fz;
            qfx[p.j] -= fx; qfy[p.j] -= fy; qfz[p.j] -= fz;
        }
        eb.pair14_lj += e_14_lj;
        eb.pair14_el += e_14_el;
    }

    // Harmonic Zone-B rim / position restraints
    if (!top.tether_k.empty() && static_cast<int32_t>(top.ref_x.size()) == top.n_atoms) {
        double e_tet = 0.0;
        for (int32_t i = 0; i < top.n_atoms; ++i) {
            const double k = top.tether_k[i];
            if (k <= 0.0) continue;
            const double dx = rx[i] - top.ref_x[i];
            const double dy = ry[i] - top.ref_y[i];
            const double dz = rz[i] - top.ref_z[i];
            e_tet += 0.5 * k * (dx * dx + dy * dy + dz * dz);
            qfx[i] += double_to_qforce(-k * dx);
            qfy[i] += double_to_qforce(-k * dy);
            qfz[i] += double_to_qforce(-k * dz);
        }
        eb.tether_buf += e_tet;
    }
}

// Linear Topological Covalent-Centroid Filter P and its adjoint P^T:
//   forward: q_tilde = P * q
//   adjoint: f_q += P^T * f_tilde
// Strictly conservative: F_q = -grad_q [ U_slow(P q) ] = -P^T grad_{q_tilde} U_slow(q_tilde)!
inline void apply_covalent_filter_positions(
    const SystemTopology& top,
    const double* __restrict rx,
    const double* __restrict ry,
    const double* __restrict rz,
    double* __restrict tx,
    double* __restrict ty,
    double* __restrict tz
) noexcept {
    const int32_t n = top.n_atoms;
    for (int32_t i = 0; i < n; ++i) {
        tx[i] = rx[i];
        ty[i] = ry[i];
        tz[i] = rz[i];
    }
    const double alpha = top.filter_alpha;
    if (alpha <= 0.0 || top.filter_groups.empty()) return;

    const double one_minus_a = 1.0 - alpha;
    for (const auto& g : top.filter_groups) {
        const int32_t p = g.heavy_idx;
        double cx = top.mass[p] * rx[p];
        double cy = top.mass[p] * ry[p];
        double cz = top.mass[p] * rz[p];
        for (int k = 0; k < g.n_hydrogens; ++k) {
            const int32_t h = g.h_indices[k];
            cx += top.mass[h] * rx[h];
            cy += top.mass[h] * ry[h];
            cz += top.mass[h] * rz[h];
        }
        cx *= g.inv_total_mass;
        cy *= g.inv_total_mass;
        cz *= g.inv_total_mass;

        tx[p] = one_minus_a * rx[p] + alpha * cx;
        ty[p] = one_minus_a * ry[p] + alpha * cy;
        tz[p] = one_minus_a * rz[p] + alpha * cz;
        for (int k = 0; k < g.n_hydrogens; ++k) {
            const int32_t h = g.h_indices[k];
            tx[h] = one_minus_a * rx[h] + alpha * cx;
            ty[h] = one_minus_a * ry[h] + alpha * cy;
            tz[h] = one_minus_a * rz[h] + alpha * cz;
        }
    }
}

// Adjoint force projection: given forces (ftx, fty, ftz) wrt filtered coordinates q_tilde,
// compute exact conservative forces wrt true coordinates q:
//   For atom u in group g:
//     q_tilde_v = (1 - a) * delta_{uv} * q_u + a * (m_u / M_g) * q_u + ...
//     => (P^T f_tilde)_u = (1 - a) * f_tilde_u + a * (m_u / M_g) * sum_{v in g} f_tilde_v
inline void apply_covalent_filter_forces_adjoint(
    const SystemTopology& top,
    double* __restrict fx,
    double* __restrict fy,
    double* __restrict fz
) noexcept {
    const double alpha = top.filter_alpha;
    if (alpha <= 0.0 || top.filter_groups.empty()) return;

    const double one_minus_a = 1.0 - alpha;
    for (const auto& g : top.filter_groups) {
        const int32_t p = g.heavy_idx;
        double sum_fx = fx[p];
        double sum_fy = fy[p];
        double sum_fz = fz[p];
        for (int k = 0; k < g.n_hydrogens; ++k) {
            const int32_t h = g.h_indices[k];
            sum_fx += fx[h];
            sum_fy += fy[h];
            sum_fz += fz[h];
        }
        const double scale_p = alpha * top.mass[p] * g.inv_total_mass;
        fx[p] = one_minus_a * fx[p] + scale_p * sum_fx;
        fy[p] = one_minus_a * fy[p] + scale_p * sum_fy;
        fz[p] = one_minus_a * fz[p] + scale_p * sum_fz;

        for (int k = 0; k < g.n_hydrogens; ++k) {
            const int32_t h = g.h_indices[k];
            const double scale_h = alpha * top.mass[h] * g.inv_total_mass;
            fx[h] = one_minus_a * fx[h] + scale_h * sum_fx;
            fy[h] = one_minus_a * fy[h] + scale_h * sum_fy;
            fz[h] = one_minus_a * fz[h] + scale_h * sum_fz;
        }
    }
}

} // namespace ashmd
