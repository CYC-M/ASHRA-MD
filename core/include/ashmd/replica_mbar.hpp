#pragma once

// Original Pillar 3: Interface Hamiltonian Replica Tempering + Exact Self-Consistent MBAR
//   - Designed specifically for Instabox 4-vCPU AMD EPYC nodes (M = 4 synchronous walkers,
//     1 physical vCPU per walker = zero lock contention!)
//   - Replica 0 is 100% UNMODIFIED physical AMBER Hamiltonian at physiological T = 310.15 K
//   - Higher replicas m = 1..M-1 temper ONLY interfacial nonbonded + peptide rotatable torsions
//     with steric soft-core smoothing, rapidly ejecting false-positive decoy poses while
//     preserving receptor fold!
//   - Exact Metropolis-Hastings coordinate swaps satisfy detailed balance rigorously
//   - Self-consistent MBAR solver computes exact unbiased 310.15 K weights W_0(x_n) and
//     thermodynamic coupling free energy Delta_F_{0 -> M-1}.

#include "integrator_drf.hpp"
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

namespace ashmd {

struct MBARAccumulator {
    int32_t n_states = 1;
    std::vector<int32_t> state_counts; // N_k samples drawn from state k
    // Flattened reduced potential matrix u_kn [size: n_samples_total * n_states],
    // where u_k(x_n) = U_k(x_n) / (k_B * T_0)
    std::vector<double> u_matrix;
    std::vector<int32_t> origin_state;

    void reset(int32_t M) {
        n_states = std::max(1, M);
        state_counts.assign(n_states, 0);
        u_matrix.clear();
        origin_state.clear();
    }

    int32_t total_samples() const noexcept {
        return static_cast<int32_t>(origin_state.size());
    }

    void record_sample(int32_t state_k, const double* u_row) {
        if (state_k < 0 || state_k >= n_states) return;
        state_counts[state_k] += 1;
        origin_state.push_back(state_k);
        u_matrix.insert(u_matrix.end(), u_row, u_row + n_states);
    }

    // Solve self-consistent MBAR equations (Shirts & Chodera 2008) for dimensionless
    // free energies f_0 = 0, f_1, ..., f_{M-1}, and normalized ground-state weights w_0[0..N-1]
    void solve_mbar(
        std::vector<double>& f_out,
        std::vector<double>& w0_out,
        int32_t max_iter = 500,
        double tol = 1.0e-9
    ) const {
        const int32_t M = n_states;
        const int32_t N = total_samples();
        f_out.assign(M, 0.0);
        w0_out.assign(N, (N > 0 ? 1.0 / static_cast<double>(N) : 0.0));
        if (N == 0 || M <= 1) return;

        std::vector<double> log_Nk(M, -1.0e30);
        for (int32_t k = 0; k < M; ++k) {
            if (state_counts[k] > 0) log_Nk[k] = std::log(static_cast<double>(state_counts[k]));
        }

        std::vector<double> log_denom(N, 0.0);
        std::vector<double> f_new(M, 0.0);

        for (int32_t iter = 0; iter < max_iter; ++iter) {
            // Compute log_denom[n] = log sum_k N_k exp(f_k - u_k(x_n))
            for (int32_t n = 0; n < N; ++n) {
                const double* u_row = &u_matrix[static_cast<size_t>(n) * M];
                double max_a = -1.0e300;
                for (int32_t k = 0; k < M; ++k) {
                    if (state_counts[k] <= 0) continue;
                    const double a = log_Nk[k] + f_out[k] - u_row[k];
                    if (a > max_a) max_a = a;
                }
                double s = 0.0;
                for (int32_t k = 0; k < M; ++k) {
                    if (state_counts[k] <= 0) continue;
                    s += std::exp((log_Nk[k] + f_out[k] - u_row[k]) - max_a);
                }
                log_denom[n] = max_a + std::log(s);
            }

            // Update f_m = -log sum_n exp(-u_m(x_n) - log_denom[n])
            for (int32_t m = 0; m < M; ++m) {
                double max_b = -1.0e300;
                for (int32_t n = 0; n < N; ++n) {
                    const double b = -u_matrix[static_cast<size_t>(n) * M + m] - log_denom[n];
                    if (b > max_b) max_b = b;
                }
                double s = 0.0;
                for (int32_t n = 0; n < N; ++n) {
                    const double b = -u_matrix[static_cast<size_t>(n) * M + m] - log_denom[n];
                    s += std::exp(b - max_b);
                }
                f_new[m] = -(max_b + std::log(s));
            }

            // Gauge fix f_0 = 0
            const double shift = f_new[0];
            double max_diff = 0.0;
            for (int32_t m = 0; m < M; ++m) {
                f_new[m] -= shift;
                const double diff = std::abs(f_new[m] - f_out[m]);
                if (diff > max_diff) max_diff = diff;
                // Damped fixed-point update ensures global convergence without overshoot
                f_out[m] = 0.7 * f_new[m] + 0.3 * f_out[m];
            }
            if (max_diff < tol) break;
        }

        // Compute normalized state-0 unbiased Boltzmann weights w0_out[n]
        double max_lw = -1.0e300;
        for (int32_t n = 0; n < N; ++n) {
            const double* u_row = &u_matrix[static_cast<size_t>(n) * M];
            double max_a = -1.0e300;
            for (int32_t k = 0; k < M; ++k) {
                if (state_counts[k] <= 0) continue;
                const double a = log_Nk[k] + f_out[k] - u_row[k];
                if (a > max_a) max_a = a;
            }
            double s = 0.0;
            for (int32_t k = 0; k < M; ++k) {
                if (state_counts[k] <= 0) continue;
                s += std::exp((log_Nk[k] + f_out[k] - u_row[k]) - max_a);
            }
            const double lw = f_out[0] - u_row[0] - (max_a + std::log(s));
            w0_out[n] = lw;
            if (lw > max_lw) max_lw = lw;
        }

        double w_sum = 0.0;
        for (int32_t n = 0; n < N; ++n) {
            w0_out[n] = std::exp(w0_out[n] - max_lw);
            w_sum += w0_out[n];
        }
        if (w_sum > 0.0) {
            const double inv_w = 1.0 / w_sum;
            for (int32_t n = 0; n < N; ++n) w0_out[n] *= inv_w;
        }
    }
};

// Helper: exact O(N_inter) fast interface direct+LJ energy at configuration of st under (lambda_inter, softcore_alpha)
inline double eval_fast_interface_energy_for_replica(
    const SystemTopology& top,
    ReplicaState& st,
    double lambda_inter,
    double softcore_alpha
) {
    if (top.solvent_mode == MODE_EXPLICIT_PME) {
        return eval_pme_interface_direct_lj_only(
            top, st.pme_ws,
            st.rx.data(), st.ry.data(), st.rz.data(),
            lambda_inter, softcore_alpha
        );
    }
    return eval_ash_interface_direct_lj_only(
        top, st.ash_ws,
        st.rx.data(), st.ry.data(), st.rz.data(),
        lambda_inter, softcore_alpha
    );
}

// Evaluate full MxM reduced potential cross-matrix u_matrix[r_cfg][m_ham] = beta_0 * U_{m_ham}(q_{r_cfg})
// Uses exact algebraic O(N_inter) decomposition so ZERO redundant full force/Born/PME evaluations occur!
inline void compute_cross_reduced_potentials(
    const SystemTopology& top,
    std::vector<ReplicaState>& replicas,
    ParallelPool& single_thread_pool,
    double target_temp_k,
    std::vector<std::vector<double>>& u_cross
) {
    const int32_t M = static_cast<int32_t>(replicas.size());
    const double beta = 1.0 / (BOLTZMANN_KCAL * std::max(target_temp_k, 1.0));
    u_cross.assign(M, std::vector<double>(M, 0.0));

    for (int32_t r = 0; r < M; ++r) {
        auto& st_r = replicas[r];
        if (!st_r.forces_initialized) {
            evaluate_all_forces(top, st_r, single_thread_pool, target_temp_k);
        }
        const auto& tp_r = st_r.tempering;
        const double raw_pep_tor  = st_r.energy_fast.raw_pep_torsion + st_r.energy_slow.raw_pep_torsion;
        const double raw_slow_int = st_r.energy_slow.raw_interface_nonb; // GB interface polarization (linear in lambda_inter, independent of softcore_alpha)
        const double e_fast_self  = eval_fast_interface_energy_for_replica(
            top, st_r, tp_r.lambda_inter, tp_r.softcore_alpha
        );
        const double u_invariant  = st_r.energy_total.potential
                                  - tp_r.mu_torsion * raw_pep_tor
                                  - tp_r.lambda_inter * raw_slow_int
                                  - e_fast_self;

        for (int32_t m = 0; m < M; ++m) {
            if (m == r) {
                u_cross[r][m] = beta * st_r.energy_total.potential;
            } else {
                const auto& tp_m = replicas[m].tempering;
                const double e_fast_m = eval_fast_interface_energy_for_replica(
                    top, st_r, tp_m.lambda_inter, tp_m.softcore_alpha
                );
                const double u_rm = u_invariant
                                  + tp_m.mu_torsion * raw_pep_tor
                                  + tp_m.lambda_inter * raw_slow_int
                                  + e_fast_m;
                u_cross[r][m] = beta * u_rm;
            }
        }
    }
}

// Attempt even-odd alternating Metropolis-Hastings swaps between adjacent IHRT replicas
inline void attempt_ihrt_swaps(
    const SystemTopology& top,
    std::vector<ReplicaState>& replicas,
    ParallelPool& single_thread_pool,
    uint64_t global_seed,
    uint64_t exchange_round,
    double target_temp_k,
    std::vector<int64_t>& swap_attempts,
    std::vector<int64_t>& swap_accepts,
    MBARAccumulator* mbar_acc = nullptr
) {
    const int32_t M = static_cast<int32_t>(replicas.size());
    if (M <= 1) return;

    std::vector<std::vector<double>> u_cross;
    compute_cross_reduced_potentials(top, replicas, single_thread_pool, target_temp_k, u_cross);

    if (mbar_acc != nullptr) {
        for (int32_t r = 0; r < M; ++r) {
            mbar_acc->record_sample(r, u_cross[r].data());
        }
    }

    const int32_t start_m = static_cast<int32_t>(exchange_round & 1ULL);
    for (int32_t m = start_m; m + 1 < M; m += 2) {
        swap_attempts[m] += 1;
        // Delta = [u_{m}(q_{m+1}) + u_{m+1}(q_m)] - [u_m(q_m) + u_{m+1}(q_{m+1})]
        const double delta = (u_cross[m + 1][m] + u_cross[m][m + 1])
                           - (u_cross[m][m] + u_cross[m + 1][m + 1]);
        bool accept = false;
        if (delta <= 0.0) {
            accept = true;
        } else {
            const double u01 = Philox4x32::uniform01(global_seed, exchange_round, static_cast<uint32_t>(m));
            accept = (u01 < std::exp(-delta));
        }

        if (accept) {
            swap_accepts[m] += 1;
            // Swap positions, velocities, AND coordinate-bound Verlet/GB workspaces
            // so neighbor lists & static scaffold integrals remain valid without rebuilding!
            std::swap(replicas[m].rx, replicas[m + 1].rx);
            std::swap(replicas[m].ry, replicas[m + 1].ry);
            std::swap(replicas[m].rz, replicas[m + 1].rz);
            std::swap(replicas[m].vx, replicas[m + 1].vx);
            std::swap(replicas[m].vy, replicas[m + 1].vy);
            std::swap(replicas[m].vz, replicas[m + 1].vz);
            std::swap(replicas[m].pme_ws, replicas[m + 1].pme_ws);
            std::swap(replicas[m].ash_ws, replicas[m + 1].ash_ws);

            evaluate_all_forces(top, replicas[m], single_thread_pool, target_temp_k);
            evaluate_all_forces(top, replicas[m + 1], single_thread_pool, target_temp_k);
        }
    }
}

} // namespace ashmd
