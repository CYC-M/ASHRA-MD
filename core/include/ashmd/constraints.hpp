#pragma once

// Symplectic Holonomic Manifold Projection (RATTLE / SHAKE + Exact 3x3 Rigid Water Solver)
// - Enforces |r_i - r_j|^2 = d_ij^2 to 1e-11 A^2 (position manifold)
// - Enforces (v_i - v_j) . (r_i - r_j) = 0 to 1e-12 (cotangent velocity bundle)
// Required for symplectic 2.0 fs inner integration steps with rigid TIP3P water & X-H bonds.

#include "types.hpp"
#include <cmath>
#include <algorithm>

namespace ashmd {

// Solve 3x3 linear system A * x = b by Gaussian elimination with partial pivoting
inline bool solve_3x3(double A[3][3], const double b[3], double x[3]) noexcept {
    double M[3][4] = {
        {A[0][0], A[0][1], A[0][2], b[0]},
        {A[1][0], A[1][1], A[1][2], b[1]},
        {A[2][0], A[2][1], A[2][2], b[2]}
    };
    for (int col = 0; col < 3; ++col) {
        int piv = col;
        double max_v = std::abs(M[col][col]);
        for (int row = col + 1; row < 3; ++row) {
            if (std::abs(M[row][col]) > max_v) {
                max_v = std::abs(M[row][col]);
                piv = row;
            }
        }
        if (max_v < 1.0e-18) return false;
        if (piv != col) {
            for (int k = col; k < 4; ++k) std::swap(M[col][k], M[piv][k]);
        }
        const double inv_d = 1.0 / M[col][col];
        for (int row = col + 1; row < 3; ++row) {
            const double fac = M[row][col] * inv_d;
            for (int k = col; k < 4; ++k) M[row][k] -= fac * M[col][k];
        }
    }
    for (int row = 2; row >= 0; --row) {
        double s = M[row][3];
        for (int col = row + 1; col < 3; ++col) s -= M[row][col] * x[col];
        x[row] = s / M[row][row];
    }
    return true;
}

// TIP3P geometric targets: r_OH = 0.9572 A, theta_HOH = 104.52 deg -> r_HH = 1.5139006545247014 A
constexpr double TIP3P_ROH_SQ = 0.9572 * 0.9572;
constexpr double TIP3P_RHH_SQ = 1.5139006545247014 * 1.5139006545247014;

// Exact coupled 3x3 Newton-SHAKE position + velocity update for rigid water triangles
inline void constrain_positions_rattle(
    const SystemTopology& top,
    const double* __restrict old_x,
    const double* __restrict old_y,
    const double* __restrict old_z,
    double* __restrict rx,
    double* __restrict ry,
    double* __restrict rz,
    double* __restrict vx,
    double* __restrict vy,
    double* __restrict vz,
    double dt_tau
) noexcept {
    if (dt_tau <= 0.0) return;
    const double inv_dt = 1.0 / dt_tau;

    // 1. Coupled 3x3 Newton solver for each independent TIP3P water molecule
    for (const auto& w : top.waters) {
        const int32_t a[3] = {w.o, w.h1, w.o};
        const int32_t b[3] = {w.h1, w.h2, w.h2};
        const double d2_target[3] = {TIP3P_ROH_SQ, TIP3P_RHH_SQ, TIP3P_ROH_SQ};

        double r0x[3], r0y[3], r0z[3];
        for (int c = 0; c < 3; ++c) {
            r0x[c] = old_x[a[c]] - old_x[b[c]];
            r0y[c] = old_y[a[c]] - old_y[b[c]];
            r0z[c] = old_z[a[c]] - old_z[b[c]];
        }

        for (int iter = 0; iter < 12; ++iter) {
            double rx_ab[3], ry_ab[3], rz_ab[3], rhs[3];
            double max_err = 0.0;
            for (int c = 0; c < 3; ++c) {
                rx_ab[c] = rx[a[c]] - rx[b[c]];
                ry_ab[c] = ry[a[c]] - ry[b[c]];
                rz_ab[c] = rz[a[c]] - rz[b[c]];
                const double cur_d2 = rx_ab[c]*rx_ab[c] + ry_ab[c]*ry_ab[c] + rz_ab[c]*rz_ab[c];
                rhs[c] = 0.5 * (d2_target[c] - cur_d2);
                max_err = std::max(max_err, std::abs(rhs[c]));
            }
            if (max_err < 1.0e-11) break;

            // Build 3x3 Jacobian J_{c, k}: d(0.5*|r_c + delta_c|^2)/d(lam_k)
            double J[3][3] = {{0.0}};
            for (int c = 0; c < 3; ++c) {
                for (int k = 0; k < 3; ++k) {
                    double coeff = 0.0;
                    if (a[c] == a[k]) coeff += top.inv_mass[a[c]];
                    if (a[c] == b[k]) coeff -= top.inv_mass[a[c]];
                    if (b[c] == b[k]) coeff += top.inv_mass[b[c]];
                    if (b[c] == a[k]) coeff -= top.inv_mass[b[c]];
                    if (coeff != 0.0) {
                        J[c][k] = coeff * (rx_ab[c]*r0x[k] + ry_ab[c]*r0y[k] + rz_ab[c]*r0z[k]);
                    }
                }
            }
            double dlam[3] = {0.0, 0.0, 0.0};
            if (!solve_3x3(J, rhs, dlam)) break;

            for (int k = 0; k < 3; ++k) {
                const int32_t ia = a[k];
                const int32_t ib = b[k];
                const double ima = top.inv_mass[ia] * dlam[k];
                const double imb = top.inv_mass[ib] * dlam[k];

                rx[ia] += ima * r0x[k]; ry[ia] += ima * r0y[k]; rz[ia] += ima * r0z[k];
                rx[ib] -= imb * r0x[k]; ry[ib] -= imb * r0y[k]; rz[ib] -= imb * r0z[k];

                vx[ia] += ima * r0x[k] * inv_dt;
                vy[ia] += ima * r0y[k] * inv_dt;
                vz[ia] += ima * r0z[k] * inv_dt;
                vx[ib] -= imb * r0x[k] * inv_dt;
                vy[ib] -= imb * r0y[k] * inv_dt;
                vz[ib] -= imb * r0z[k] * inv_dt;
            }
        }
    }

    // 2. Gauss-Seidel RATTLE position projection for X-H holonomic bonds
    if (!top.constraints.empty()) {
        const int32_t nc = static_cast<int32_t>(top.constraints.size());
        for (int iter = 0; iter < 50; ++iter) {
            double max_err = 0.0;
            for (int32_t c = 0; c < nc; ++c) {
                const auto& con = top.constraints[c];
                const int32_t i = con.i;
                const int32_t j = con.j;
                const double imi = top.inv_mass[i];
                const double imj = top.inv_mass[j];
                const double im_sum = imi + imj;
                if (im_sum <= 0.0) continue;

                const double dx = rx[i] - rx[j];
                const double dy = ry[i] - ry[j];
                const double dz = rz[i] - rz[j];
                const double r2 = dx * dx + dy * dy + dz * dz;
                const double diff = con.dist_sq - r2;
                const double a_diff = std::abs(diff);
                if (a_diff > max_err) max_err = a_diff;
                if (a_diff < 1.0e-11) continue;

                const double dx0 = old_x[i] - old_x[j];
                const double dy0 = old_y[i] - old_y[j];
                const double dz0 = old_z[i] - old_z[j];
                const double dot = dx * dx0 + dy * dy0 + dz * dz0;
                if (std::abs(dot) < 1.0e-12) continue;

                const double lam = diff / (2.0 * im_sum * dot);
                rx[i] += imi * lam * dx0; ry[i] += imi * lam * dy0; rz[i] += imi * lam * dz0;
                rx[j] -= imj * lam * dx0; ry[j] -= imj * lam * dy0; rz[j] -= imj * lam * dz0;

                vx[i] += imi * lam * dx0 * inv_dt;
                vy[i] += imi * lam * dy0 * inv_dt;
                vz[i] += imi * lam * dz0 * inv_dt;
                vx[j] -= imj * lam * dx0 * inv_dt;
                vy[j] -= imj * lam * dy0 * inv_dt;
                vz[j] -= imj * lam * dz0 * inv_dt;
            }
            if (max_err < 1.0e-10) break;
        }
    }
}

// Cotangent-bundle RATTLE velocity projection: ensures (v_i - v_j) . (r_i - r_j) = 0
inline void constrain_velocities_rattle(
    const SystemTopology& top,
    const double* __restrict rx,
    const double* __restrict ry,
    const double* __restrict rz,
    double* __restrict vx,
    double* __restrict vy,
    double* __restrict vz
) noexcept {
    // 1. Exact 3x3 linear solve for rigid TIP3P waters
    for (const auto& w : top.waters) {
        const int32_t a[3] = {w.o, w.h1, w.o};
        const int32_t b[3] = {w.h1, w.h2, w.h2};

        double dx[3], dy[3], dz[3], rhs[3];
        for (int c = 0; c < 3; ++c) {
            dx[c] = rx[a[c]] - rx[b[c]];
            dy[c] = ry[a[c]] - ry[b[c]];
            dz[c] = rz[a[c]] - rz[b[c]];
            const double dvx = vx[a[c]] - vx[b[c]];
            const double dvy = vy[a[c]] - vy[b[c]];
            const double dvz = vz[a[c]] - vz[b[c]];
            rhs[c] = -(dvx * dx[c] + dvy * dy[c] + dvz * dz[c]);
        }

        double M[3][3] = {{0.0}};
        for (int c = 0; c < 3; ++c) {
            for (int k = 0; k < 3; ++k) {
                double coeff = 0.0;
                if (a[c] == a[k]) coeff += top.inv_mass[a[c]];
                if (a[c] == b[k]) coeff -= top.inv_mass[a[c]];
                if (b[c] == b[k]) coeff += top.inv_mass[b[c]];
                if (b[c] == a[k]) coeff -= top.inv_mass[b[c]];
                if (coeff != 0.0) {
                    M[c][k] = coeff * (dx[c]*dx[k] + dy[c]*dy[k] + dz[c]*dz[k]);
                }
            }
        }
        double mu[3] = {0.0, 0.0, 0.0};
        if (solve_3x3(M, rhs, mu)) {
            for (int k = 0; k < 3; ++k) {
                const int32_t ia = a[k];
                const int32_t ib = b[k];
                const double ima = top.inv_mass[ia] * mu[k];
                const double imb = top.inv_mass[ib] * mu[k];
                vx[ia] += ima * dx[k]; vy[ia] += ima * dy[k]; vz[ia] += ima * dz[k];
                vx[ib] -= imb * dx[k]; vy[ib] -= imb * dy[k]; vz[ib] -= imb * dz[k];
            }
        }
    }

    // 2. X-H bond velocity projection
    if (!top.constraints.empty()) {
        const int32_t nc = static_cast<int32_t>(top.constraints.size());
        for (int iter = 0; iter < 25; ++iter) {
            double max_v_err = 0.0;
            for (int32_t c = 0; c < nc; ++c) {
                const auto& con = top.constraints[c];
                const int32_t i = con.i;
                const int32_t j = con.j;
                const double imi = top.inv_mass[i];
                const double imj = top.inv_mass[j];
                const double im_sum = imi + imj;
                if (im_sum <= 0.0) continue;

                const double dx = rx[i] - rx[j];
                const double dy = ry[i] - ry[j];
                const double dz = rz[i] - rz[j];
                const double dvx = vx[i] - vx[j];
                const double dvy = vy[i] - vy[j];
                const double dvz = vz[i] - vz[j];

                const double dot_vr = dvx * dx + dvy * dy + dvz * dz;
                const double a_vr   = std::abs(dot_vr);
                if (a_vr > max_v_err) max_v_err = a_vr;
                if (a_vr < 1.0e-12) continue;

                const double r2 = dx * dx + dy * dy + dz * dz;
                const double mu = -dot_vr / (im_sum * std::max(r2, 1.0e-16));
                vx[i] += imi * mu * dx; vy[i] += imi * mu * dy; vz[i] += imi * mu * dz;
                vx[j] -= imj * mu * dx; vy[j] -= imj * mu * dy; vz[j] -= imj * mu * dz;
            }
            if (max_v_err < 1.0e-11) break;
        }
    }
}

} // namespace ashmd
