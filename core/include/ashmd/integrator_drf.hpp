#pragma once

// Original Pillar 1: Deterministic Resonance-Free Split Langevin Integrator (FC-BAOAB)
//   - Linear Topological Covalent-Centroid Filter P suppresses X-H / outer-step resonance
//     while remaining 100% conservative: F_slow = -grad_q [ U_slow(P q) ] = -P^T grad U_slow(P q)
//   - Geodesic BAOAB splitting on holonomic constraint manifold (SETTLE + RATTLE)
//   - Philox-4x32 counter-based stochastic thermostat (bit-identical across 1..N threads & checkpoints)
//   - Q31.32 associative fixed-point force accumulation (zero OpenMP summation drift)

#include "types.hpp"
#include "philox_rng.hpp"
#include "parallel.hpp"
#include "bonded.hpp"
#include "constraints.hpp"
#include "nonbonded_pme.hpp"
#include "active_shell_gb.hpp"

#include <cmath>
#include <vector>
#include <algorithm>

namespace ashmd {

struct ReplicaState {
    uint32_t replica_id = 0;
    ReplicaTemperingParams tempering;

    std::vector<double> rx, ry, rz;     // Positions [A]
    std::vector<double> vx, vy, vz;     // Internal velocities [sqrt(kcal/mol/amu)]

    std::vector<double> fx_fast, fy_fast, fz_fast;
    std::vector<double> fx_slow, fy_slow, fz_slow;
    std::vector<double> tx, ty, tz;     // Filtered coordinates P*q
    std::vector<double> old_x, old_y, old_z;
    std::vector<int64_t> qfx, qfy, qfz;

    EnergyBreakdown energy_fast;
    EnergyBreakdown energy_slow;
    EnergyBreakdown energy_total;

    PMEWorkspace pme_ws;
    ASHWorkspace ash_ws;
    bool forces_initialized = false;

    void allocate(int32_t n_atoms) {
        rx.assign(n_atoms, 0.0); ry.assign(n_atoms, 0.0); rz.assign(n_atoms, 0.0);
        vx.assign(n_atoms, 0.0); vy.assign(n_atoms, 0.0); vz.assign(n_atoms, 0.0);
        fx_fast.assign(n_atoms, 0.0); fy_fast.assign(n_atoms, 0.0); fz_fast.assign(n_atoms, 0.0);
        fx_slow.assign(n_atoms, 0.0); fy_slow.assign(n_atoms, 0.0); fz_slow.assign(n_atoms, 0.0);
        tx.assign(n_atoms, 0.0); ty.assign(n_atoms, 0.0); tz.assign(n_atoms, 0.0);
        old_x.assign(n_atoms, 0.0); old_y.assign(n_atoms, 0.0); old_z.assign(n_atoms, 0.0);
        qfx.assign(n_atoms, 0); qfy.assign(n_atoms, 0); qfz.assign(n_atoms, 0);
        forces_initialized = false;
    }
};

inline void compute_kinetic_and_temp(
    const SystemTopology& top,
    const ReplicaState& st,
    double& ek_out,
    double& temp_out
) noexcept {
    double ek = 0.0;
    for (int32_t i = 0; i < top.n_atoms; ++i) {
        if (top.inv_mass[i] <= 0.0) continue;
        const double v2 = st.vx[i]*st.vx[i] + st.vy[i]*st.vy[i] + st.vz[i]*st.vz[i];
        ek += 0.5 * top.mass[i] * v2;
    }
    ek_out = ek;
    const int32_t dof = top.degrees_of_freedom();
    temp_out = (2.0 * ek) / (static_cast<double>(dof) * BOLTZMANN_KCAL);
}

// Initialize Maxwell-Boltzmann velocities at target_temp_k using deterministic Philox RNG
inline void initialize_velocities(
    const SystemTopology& top,
    ReplicaState& st,
    uint64_t seed,
    double target_temp_k
) noexcept {
    const int32_t n = top.n_atoms;
    double px = 0.0, py = 0.0, pz = 0.0;
    double m_tot = 0.0;
    bool remove_com = true;

    for (int32_t i = 0; i < n; ++i) {
        if (top.inv_mass[i] <= 0.0 || top.tether_k[i] > 0.0) remove_com = false;
        if (top.inv_mass[i] <= 0.0) {
            st.vx[i] = st.vy[i] = st.vz[i] = 0.0;
            continue;
        }
        double gx, gy, gz;
        Philox4x32::normal3(seed ^ 0xA5A55A5AULL, st.replica_id, 0ULL, static_cast<uint32_t>(i), gx, gy, gz);
        const double sigma = std::sqrt(BOLTZMANN_KCAL * target_temp_k * top.inv_mass[i]);
        st.vx[i] = sigma * gx;
        st.vy[i] = sigma * gy;
        st.vz[i] = sigma * gz;
        px += top.mass[i] * st.vx[i];
        py += top.mass[i] * st.vy[i];
        pz += top.mass[i] * st.vz[i];
        m_tot += top.mass[i];
    }

    if (remove_com && m_tot > 0.0) {
        const double vx_com = px / m_tot;
        const double vy_com = py / m_tot;
        const double vz_com = pz / m_tot;
        for (int32_t i = 0; i < n; ++i) {
            if (top.inv_mass[i] > 0.0) {
                st.vx[i] -= vx_com;
                st.vy[i] -= vy_com;
                st.vz[i] -= vz_com;
            }
        }
    }

    constrain_velocities_rattle(top, st.rx.data(), st.ry.data(), st.rz.data(),
                                st.vx.data(), st.vy.data(), st.vz.data());

    // Rescale once after projection so initial kinetic temperature matches target_temp_k
    if (target_temp_k > 0.0) {
        double ek = 0.0, t_cur = 0.0;
        compute_kinetic_and_temp(top, st, ek, t_cur);
        if (t_cur > 1.0e-8) {
            const double s = std::sqrt(target_temp_k / t_cur);
            for (int32_t i = 0; i < n; ++i) {
                st.vx[i] *= s; st.vy[i] *= s; st.vz[i] *= s;
            }
        }
    }
}

// Evaluate Fast + Direct Nonbonded Forces
inline void evaluate_fast_forces(
    const SystemTopology& top,
    ReplicaState& st,
    ParallelPool& pool,
    double target_temp_k
) {
    const int32_t n = top.n_atoms;
    std::fill(st.qfx.begin(), st.qfx.end(), 0);
    std::fill(st.qfy.begin(), st.qfy.end(), 0);
    std::fill(st.qfz.begin(), st.qfz.end(), 0);
    st.energy_fast = EnergyBreakdown{};

    compute_bonded_and_tether(
        top, st.rx.data(), st.ry.data(), st.rz.data(),
        st.tempering, st.qfx.data(), st.qfy.data(), st.qfz.data(), st.energy_fast, true
    );

    if (top.solvent_mode == MODE_EXPLICIT_PME) {
        compute_pme_direct_and_lj(
            top, st.pme_ws, pool,
            st.rx.data(), st.ry.data(), st.rz.data(),
            st.tempering, st.qfx.data(), st.qfy.data(), st.qfz.data(), st.energy_fast
        );
    } else {
        compute_ash_direct_and_cap(
            top, st.ash_ws, pool,
            st.rx.data(), st.ry.data(), st.rz.data(),
            st.tempering, st.qfx.data(), st.qfy.data(), st.qfz.data(), st.energy_fast, target_temp_k
        );
    }

    for (int32_t i = 0; i < n; ++i) {
        if (top.inv_mass[i] <= 0.0) {
            st.fx_fast[i] = st.fy_fast[i] = st.fz_fast[i] = 0.0;
        } else {
            st.fx_fast[i] = qforce_to_double(st.qfx[i]);
            st.fy_fast[i] = qforce_to_double(st.qfy[i]);
            st.fz_fast[i] = qforce_to_double(st.qfz[i]);
        }
    }
}

// Evaluate Slow Long-Range / Continuum Solvation Forces on Filtered Coordinates q_tilde = P * q
inline void evaluate_slow_forces(
    const SystemTopology& top,
    ReplicaState& st,
    ParallelPool& pool,
    double target_temp_k
) {
    const int32_t n = top.n_atoms;
    std::fill(st.qfx.begin(), st.qfx.end(), 0);
    std::fill(st.qfy.begin(), st.qfy.end(), 0);
    std::fill(st.qfz.begin(), st.qfz.end(), 0);
    st.energy_slow = EnergyBreakdown{};

    apply_covalent_filter_positions(
        top, st.rx.data(), st.ry.data(), st.rz.data(),
        st.tx.data(), st.ty.data(), st.tz.data()
    );

    if (top.solvent_mode == MODE_EXPLICIT_PME) {
        compute_pme_reciprocal(
            top, st.pme_ws,
            st.tx.data(), st.ty.data(), st.tz.data(),
            st.qfx.data(), st.qfy.data(), st.qfz.data(), st.energy_slow
        );
    } else {
        compute_ash_gb_and_nonpolar(
            top, st.ash_ws, pool,
            st.tx.data(), st.ty.data(), st.tz.data(),
            st.tempering, st.qfx.data(), st.qfy.data(), st.qfz.data(), st.energy_slow, target_temp_k
        );
    }

    for (int32_t i = 0; i < n; ++i) {
        st.fx_slow[i] = qforce_to_double(st.qfx[i]);
        st.fy_slow[i] = qforce_to_double(st.qfy[i]);
        st.fz_slow[i] = qforce_to_double(st.qfz[i]);
    }

    // Exact conservative adjoint P^T projection onto atomic coordinates q
    apply_covalent_filter_forces_adjoint(top, st.fx_slow.data(), st.fy_slow.data(), st.fz_slow.data());

    for (int32_t i = 0; i < n; ++i) {
        if (top.inv_mass[i] <= 0.0) {
            st.fx_slow[i] = st.fy_slow[i] = st.fz_slow[i] = 0.0;
        }
    }
}

inline void combine_energies(
    const SystemTopology& top,
    ReplicaState& st
) noexcept {
    EnergyBreakdown& tot = st.energy_total;
    const auto& f = st.energy_fast;
    const auto& s = st.energy_slow;

    tot.bond       = f.bond + s.bond;
    tot.angle      = f.angle + s.angle;
    tot.dihedral   = f.dihedral + s.dihedral;
    tot.pair14_lj  = f.pair14_lj + s.pair14_lj;
    tot.pair14_el  = f.pair14_el + s.pair14_el;
    tot.lj_sr      = f.lj_sr + s.lj_sr;
    tot.lj_disp    = f.lj_disp + s.lj_disp;
    tot.elec_dir   = f.elec_dir + s.elec_dir;
    tot.elec_recip = f.elec_recip + s.elec_recip;
    tot.elec_self  = f.elec_self + s.elec_self;
    tot.elec_excl  = f.elec_excl + s.elec_excl;
    tot.gb_pol     = f.gb_pol + s.gb_pol;
    tot.np_sasa    = f.np_sasa + s.np_sasa;
    tot.cap_shell  = f.cap_shell + s.cap_shell;
    tot.tether_buf = f.tether_buf + s.tether_buf;
    tot.raw_pep_torsion    = f.raw_pep_torsion + s.raw_pep_torsion;
    tot.raw_interface_nonb = f.raw_interface_nonb + s.raw_interface_nonb;

    compute_kinetic_and_temp(top, st, tot.kinetic, tot.temperature);
    tot.finalize_potential();
}

// Single-point total energy & total force evaluation (used for gradient tests, EM, and exact MBAR cross-energies)
inline void evaluate_all_forces(
    const SystemTopology& top,
    ReplicaState& st,
    ParallelPool& pool,
    double target_temp_k = PHYSIOLOGICAL_TEMP_K
) {
    evaluate_fast_forces(top, st, pool, target_temp_k);
    evaluate_slow_forces(top, st, pool, target_temp_k);
    combine_energies(top, st);
    st.forces_initialized = true;
}

// Advance 1 outer step (= mts_ratio inner steps of dt_ps each) using FC-BAOAB
inline void step_fc_baoab_outer(
    const SystemTopology& top,
    ReplicaState& st,
    ParallelPool& pool,
    uint64_t global_seed,
    uint64_t& inner_step_counter,
    double dt_ps,
    int32_t mts_ratio,
    double gamma_ps_inv,
    double target_temp_k
) {
    const int32_t n = top.n_atoms;
    const int32_t M = std::max(1, mts_ratio);
    const double dtau_in  = dt_ps * TIME_FACTOR;
    const double dtau_out = static_cast<double>(M) * dtau_in;
    const double half_in  = 0.5 * dtau_in;
    const double half_out = 0.5 * dtau_out;

    if (!st.forces_initialized) {
        evaluate_all_forces(top, st, pool, target_temp_k);
    }

    // 1. Outer B half-kick (slow forces)
    for (int32_t i = 0; i < n; ++i) {
        const double im = top.inv_mass[i];
        if (im <= 0.0) continue;
        st.vx[i] += half_out * im * st.fx_slow[i];
        st.vy[i] += half_out * im * st.fy_slow[i];
        st.vz[i] += half_out * im * st.fz_slow[i];
    }
    constrain_velocities_rattle(top, st.rx.data(), st.ry.data(), st.rz.data(),
                                st.vx.data(), st.vy.data(), st.vz.data());

    const bool use_thermostat = (gamma_ps_inv > 0.0 && target_temp_k > 0.0);
    const double c1 = use_thermostat ? std::exp(-gamma_ps_inv * dt_ps) : 1.0;
    const double c2 = use_thermostat ? std::sqrt(std::max(0.0, 1.0 - c1 * c1)) : 0.0;

    // 2. M inner BAOAB steps on fast + short-range forces
    for (int32_t m = 0; m < M; ++m) {
        // Inner B (half-kick)
        for (int32_t i = 0; i < n; ++i) {
            const double im = top.inv_mass[i];
            if (im <= 0.0) continue;
            st.vx[i] += half_in * im * st.fx_fast[i];
            st.vy[i] += half_in * im * st.fy_fast[i];
            st.vz[i] += half_in * im * st.fz_fast[i];
        }
        constrain_velocities_rattle(top, st.rx.data(), st.ry.data(), st.rz.data(),
                                    st.vx.data(), st.vy.data(), st.vz.data());

        // Inner A (first half-drift + RATTLE position projection)
        for (int32_t i = 0; i < n; ++i) {
            st.old_x[i] = st.rx[i]; st.old_y[i] = st.ry[i]; st.old_z[i] = st.rz[i];
            if (top.inv_mass[i] > 0.0) {
                st.rx[i] += half_in * st.vx[i];
                st.ry[i] += half_in * st.vy[i];
                st.rz[i] += half_in * st.vz[i];
            }
        }
        constrain_positions_rattle(top, st.old_x.data(), st.old_y.data(), st.old_z.data(),
                                   st.rx.data(), st.ry.data(), st.rz.data(),
                                   st.vx.data(), st.vy.data(), st.vz.data(), half_in);
        constrain_velocities_rattle(top, st.rx.data(), st.ry.data(), st.rz.data(),
                                    st.vx.data(), st.vy.data(), st.vz.data());

        // Inner O (exact Ornstein-Uhlenbeck stochastic step with deterministic Philox-4x32)
        if (use_thermostat) {
            const uint64_t cur_step = inner_step_counter;
            for (int32_t i = 0; i < n; ++i) {
                const double im = top.inv_mass[i];
                if (im <= 0.0) continue;
                double gx, gy, gz;
                Philox4x32::normal3(global_seed, st.replica_id, cur_step, static_cast<uint32_t>(i), gx, gy, gz);
                const double noise_amp = c2 * std::sqrt(BOLTZMANN_KCAL * target_temp_k * im);
                st.vx[i] = c1 * st.vx[i] + noise_amp * gx;
                st.vy[i] = c1 * st.vy[i] + noise_amp * gy;
                st.vz[i] = c1 * st.vz[i] + noise_amp * gz;
            }
            constrain_velocities_rattle(top, st.rx.data(), st.ry.data(), st.rz.data(),
                                        st.vx.data(), st.vy.data(), st.vz.data());
        }

        // Inner A (second half-drift + RATTLE position projection)
        for (int32_t i = 0; i < n; ++i) {
            st.old_x[i] = st.rx[i]; st.old_y[i] = st.ry[i]; st.old_z[i] = st.rz[i];
            if (top.inv_mass[i] > 0.0) {
                st.rx[i] += half_in * st.vx[i];
                st.ry[i] += half_in * st.vy[i];
                st.rz[i] += half_in * st.vz[i];
            }
        }
        constrain_positions_rattle(top, st.old_x.data(), st.old_y.data(), st.old_z.data(),
                                   st.rx.data(), st.ry.data(), st.rz.data(),
                                   st.vx.data(), st.vy.data(), st.vz.data(), half_in);
        constrain_velocities_rattle(top, st.rx.data(), st.ry.data(), st.rz.data(),
                                    st.vx.data(), st.vy.data(), st.vz.data());

        // Evaluate fast forces at new configuration
        evaluate_fast_forces(top, st, pool, target_temp_k);

        // Inner B (second half-kick)
        for (int32_t i = 0; i < n; ++i) {
            const double im = top.inv_mass[i];
            if (im <= 0.0) continue;
            st.vx[i] += half_in * im * st.fx_fast[i];
            st.vy[i] += half_in * im * st.fy_fast[i];
            st.vz[i] += half_in * im * st.fz_fast[i];
        }
        constrain_velocities_rattle(top, st.rx.data(), st.ry.data(), st.rz.data(),
                                    st.vx.data(), st.vy.data(), st.vz.data());

        ++inner_step_counter;
    }

    // 3. Re-evaluate slow forces on filtered coordinates & apply second outer B half-kick
    evaluate_slow_forces(top, st, pool, target_temp_k);
    for (int32_t i = 0; i < n; ++i) {
        const double im = top.inv_mass[i];
        if (im <= 0.0) continue;
        st.vx[i] += half_out * im * st.fx_slow[i];
        st.vy[i] += half_out * im * st.fy_slow[i];
        st.vz[i] += half_out * im * st.fz_slow[i];
    }
    constrain_velocities_rattle(top, st.rx.data(), st.ry.data(), st.rz.data(),
                                st.vx.data(), st.vy.data(), st.vz.data());

    // Wrap rigid water molecules cleanly into primary periodic cell in PME mode (preserving O-H vectors!)
    if (top.solvent_mode == MODE_EXPLICIT_PME && top.box_x > 0.0) {
        for (const auto& w : top.waters) {
            const double sx = top.box_x * std::floor(st.rx[w.o] / top.box_x);
            const double sy = top.box_y * std::floor(st.ry[w.o] / top.box_y);
            const double sz = top.box_z * std::floor(st.rz[w.o] / top.box_z);
            if (sx != 0.0 || sy != 0.0 || sz != 0.0) {
                st.rx[w.o]  -= sx; st.ry[w.o]  -= sy; st.rz[w.o]  -= sz;
                st.rx[w.h1] -= sx; st.ry[w.h1] -= sy; st.rz[w.h1] -= sz;
                st.rx[w.h2] -= sx; st.ry[w.h2] -= sy; st.rz[w.h2] -= sz;
            }
        }
    }

    combine_energies(top, st);
}

// Step-limited adaptive Steepest Descent energy minimizer with holonomic projection
inline int32_t minimize_energy_sd(
    const SystemTopology& top,
    ReplicaState& st,
    ParallelPool& pool,
    int32_t max_steps,
    double fmax_tol = 10.0,
    double init_step_a = 0.01
) {
    const int32_t n = top.n_atoms;
    // Temporarily disable covalent filter during minimization so exact atomic clashes relax
    SystemTopology min_top = top;
    min_top.filter_alpha = 0.0;

    evaluate_all_forces(min_top, st, pool, PHYSIOLOGICAL_TEMP_K);
    double step_size = init_step_a;
    int32_t steps_done = 0;

    std::vector<double> bak_x(n), bak_y(n), bak_z(n);
    std::vector<double> dummy_vx(n, 0.0), dummy_vy(n, 0.0), dummy_vz(n, 0.0);

    for (int32_t step = 0; step < max_steps; ++step) {
        double max_f = 0.0;
        for (int32_t i = 0; i < n; ++i) {
            if (min_top.inv_mass[i] <= 0.0) continue;
            const double fx = st.fx_fast[i] + st.fx_slow[i];
            const double fy = st.fy_fast[i] + st.fy_slow[i];
            const double fz = st.fz_fast[i] + st.fz_slow[i];
            const double fm = std::sqrt(fx*fx + fy*fy + fz*fz);
            if (fm > max_f) max_f = fm;
        }
        if (max_f < fmax_tol || max_f <= 1.0e-12) break;

        const double e_prev = st.energy_total.potential;
        bak_x = st.rx; bak_y = st.ry; bak_z = st.rz;

        const double scale = step_size / max_f;
        for (int32_t i = 0; i < n; ++i) {
            if (min_top.inv_mass[i] <= 0.0) continue;
            const double fx = st.fx_fast[i] + st.fx_slow[i];
            const double fy = st.fy_fast[i] + st.fy_slow[i];
            const double fz = st.fz_fast[i] + st.fz_slow[i];
            st.rx[i] += scale * fx;
            st.ry[i] += scale * fy;
            st.rz[i] += scale * fz;
        }
        std::fill(dummy_vx.begin(), dummy_vx.end(), 0.0);
        std::fill(dummy_vy.begin(), dummy_vy.end(), 0.0);
        std::fill(dummy_vz.begin(), dummy_vz.end(), 0.0);
        constrain_positions_rattle(min_top, bak_x.data(), bak_y.data(), bak_z.data(),
                                   st.rx.data(), st.ry.data(), st.rz.data(),
                                   dummy_vx.data(), dummy_vy.data(), dummy_vz.data(), 1.0);

        evaluate_all_forces(min_top, st, pool, PHYSIOLOGICAL_TEMP_K);
        if (st.energy_total.potential < e_prev) {
            step_size = std::min(0.05, step_size * 1.2);
            ++steps_done;
        } else {
            st.rx = bak_x; st.ry = bak_y; st.rz = bak_z;
            evaluate_all_forces(min_top, st, pool, PHYSIOLOGICAL_TEMP_K);
            step_size *= 0.5;
            if (step_size < 1.0e-7) break;
        }
    }

    // Refresh forces with true production topology (including FC-BAOAB filter)
    evaluate_all_forces(top, st, pool, PHYSIOLOGICAL_TEMP_K);
    return steps_done;
}

} // namespace ashmd
