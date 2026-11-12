#include "../include/ashmd/types.hpp"
#include "../include/ashmd/simd_port.hpp"
#include "../include/ashmd/parallel.hpp"
#include "../include/ashmd/integrator_drf.hpp"
#include "../include/ashmd/replica_mbar.hpp"
#include "../include/ashmd/checkpoint.hpp"

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#if defined(_WIN32)
    #define ASHMD_EXPORT extern "C" __declspec(dllexport)
#else
    #define ASHMD_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {

struct EngineHandle {
    ashmd::SystemTopology top;
    std::unique_ptr<ashmd::ParallelPool> main_pool;
    std::unique_ptr<ashmd::ParallelPool> serial_pool;
    std::vector<std::unique_ptr<ashmd::ParallelPool>> rep_pools;

    std::vector<ashmd::ReplicaState> replicas;
    std::vector<uint64_t> inner_step_counters;
    uint64_t global_seed = 42ULL;
    uint64_t outer_step_done = 0ULL;
    uint64_t exchange_round = 0ULL;

    std::vector<int64_t> swap_attempts;
    std::vector<int64_t> swap_accepts;
    ashmd::MBARAccumulator mbar_acc;

    std::string last_error;

    explicit EngineHandle(int32_t n_threads)
        : main_pool(std::make_unique<ashmd::ParallelPool>(std::max(1, n_threads))),
          serial_pool(std::make_unique<ashmd::ParallelPool>(1)) {}

    void setup_replicas(
        int32_t n_rep,
        const double* lambda_inter,
        const double* mu_torsion,
        const double* softcore_alpha,
        uint64_t seed
    ) {
        n_rep = std::max(1, n_rep);
        global_seed = seed;
        outer_step_done = 0ULL;
        exchange_round = 0ULL;

        replicas.resize(n_rep);
        inner_step_counters.assign(n_rep, 0ULL);
        rep_pools.clear();
        const int32_t inner_threads = (n_rep == 1) ? main_pool->num_threads() : 1;
        for (int32_t r = 0; r < n_rep; ++r) {
            replicas[r].replica_id = static_cast<uint32_t>(r);
            replicas[r].allocate(top.n_atoms);
            if (lambda_inter && mu_torsion && softcore_alpha) {
                replicas[r].tempering.lambda_inter   = lambda_inter[r];
                replicas[r].tempering.mu_torsion     = mu_torsion[r];
                replicas[r].tempering.softcore_alpha = softcore_alpha[r];
            } else {
                replicas[r].tempering = {1.0, 1.0, 0.0};
            }
            rep_pools.push_back(std::make_unique<ashmd::ParallelPool>(inner_threads));
        }
        swap_attempts.assign(std::max(0, n_rep - 1), 0);
        swap_accepts.assign(std::max(0, n_rep - 1), 0);
        mbar_acc.reset(n_rep);
    }
};

} // namespace

// Plain-old-data C struct matching EnergyBreakdown layout for ctypes
struct AshmdEnergyC {
    double bond;
    double angle;
    double dihedral;
    double pair14_lj;
    double pair14_el;
    double lj_sr;
    double lj_disp;
    double elec_dir;
    double elec_recip;
    double elec_self;
    double elec_excl;
    double gb_pol;
    double np_sasa;
    double cap_shell;
    double tether_buf;
    double potential;
    double kinetic;
    double total;
    double temperature;
    double raw_pep_torsion;
    double raw_interface_nonb;
};

static void copy_energy_to_c(const ashmd::EnergyBreakdown& src, AshmdEnergyC* dst) noexcept {
    if (!dst) return;
    dst->bond       = src.bond;
    dst->angle      = src.angle;
    dst->dihedral   = src.dihedral;
    dst->pair14_lj  = src.pair14_lj;
    dst->pair14_el  = src.pair14_el;
    dst->lj_sr      = src.lj_sr;
    dst->lj_disp    = src.lj_disp;
    dst->elec_dir   = src.elec_dir;
    dst->elec_recip = src.elec_recip;
    dst->elec_self  = src.elec_self;
    dst->elec_excl  = src.elec_excl;
    dst->gb_pol     = src.gb_pol;
    dst->np_sasa    = src.np_sasa;
    dst->cap_shell  = src.cap_shell;
    dst->tether_buf = src.tether_buf;
    dst->potential  = src.potential;
    dst->kinetic    = src.kinetic;
    dst->total      = src.total;
    dst->temperature = src.temperature;
    dst->raw_pep_torsion    = src.raw_pep_torsion;
    dst->raw_interface_nonb = src.raw_interface_nonb;
}

ASHMD_EXPORT const char* ashmd_simd_backend() {
    return ashmd::simd::backend_name();
}

ASHMD_EXPORT void* ashmd_create_engine(int32_t n_threads) {
    return new EngineHandle(n_threads);
}

ASHMD_EXPORT void ashmd_destroy_engine(void* ptr) {
    delete static_cast<EngineHandle*>(ptr);
}

ASHMD_EXPORT const char* ashmd_last_error(void* ptr) {
    auto* h = static_cast<EngineHandle*>(ptr);
    return h ? h->last_error.c_str() : "Null engine handle";
}

ASHMD_EXPORT int32_t ashmd_num_replicas(void* ptr) {
    auto* h = static_cast<EngineHandle*>(ptr);
    return h ? static_cast<int32_t>(h->replicas.size()) : 0;
}

ASHMD_EXPORT int32_t ashmd_set_atoms(
    void* ptr,
    int32_t n_atoms,
    int32_t solvent_mode,
    const double* mass,
    const double* charge,
    const double* rmin_half,
    const double* epsilon,
    const double* gb_radius,
    const double* gb_screen,
    const int32_t* zone,
    const int32_t* role
) {
    auto* h = static_cast<EngineHandle*>(ptr);
    if (!h || n_atoms <= 0) return -1;

    auto& t = h->top;
    t = ashmd::SystemTopology{};
    t.n_atoms = n_atoms;
    t.solvent_mode = (solvent_mode == 0) ? ashmd::MODE_EXPLICIT_PME : ashmd::MODE_ACTIVE_SHELL;

    t.mass.assign(mass, mass + n_atoms);
    t.inv_mass.resize(n_atoms, 0.0);
    for (int32_t i = 0; i < n_atoms; ++i) {
        if (zone[i] == ashmd::ZONE_SCAFFOLD || mass[i] <= 0.0) {
            t.inv_mass[i] = 0.0;
        } else {
            t.inv_mass[i] = 1.0 / mass[i];
        }
    }
    t.charge.assign(charge, charge + n_atoms);
    t.rmin_half.assign(rmin_half, rmin_half + n_atoms);
    t.epsilon.assign(epsilon, epsilon + n_atoms);
    t.gb_radius.assign(gb_radius, gb_radius + n_atoms);
    t.gb_screen.assign(gb_screen, gb_screen + n_atoms);
    t.res_diel.assign(n_atoms, 2.0);
    t.zone.assign(zone, zone + n_atoms);
    t.role.assign(role, role + n_atoms);

    t.ref_x.assign(n_atoms, 0.0);
    t.ref_y.assign(n_atoms, 0.0);
    t.ref_z.assign(n_atoms, 0.0);
    t.tether_k.assign(n_atoms, 0.0);
    t.excl_offsets.assign(n_atoms + 1, 0);
    return 0;
}

ASHMD_EXPORT int32_t ashmd_set_res_diel(
    void* ptr,
    const double* res_diel
) {
    auto* h = static_cast<EngineHandle*>(ptr);
    if (!h || !res_diel || h->top.n_atoms <= 0) return -1;
    h->top.res_diel.assign(res_diel, res_diel + h->top.n_atoms);
    return 0;
}

ASHMD_EXPORT int32_t ashmd_set_nonbonded_params(
    void* ptr,
    double box_x, double box_y, double box_z,
    double cutoff, double switch_on, double ewald_alpha,
    int32_t pme_kx, int32_t pme_ky, int32_t pme_kz,
    double gb_cutoff, double eps_in, double eps_out,
    double ionic_strength, double sasa_gamma,
    double filter_alpha
) {
    auto* h = static_cast<EngineHandle*>(ptr);
    if (!h) return -1;
    auto& t = h->top;
    t.box_x = box_x; t.box_y = box_y; t.box_z = box_z;
    t.cutoff = cutoff; t.switch_on = switch_on; t.ewald_alpha = ewald_alpha;
    t.pme_grid_x = pme_kx; t.pme_grid_y = pme_ky; t.pme_grid_z = pme_kz;
    t.gb_cutoff = gb_cutoff; t.eps_in = eps_in; t.eps_out = eps_out;
    t.ionic_strength = ionic_strength; t.sasa_gamma = sasa_gamma;
    t.filter_alpha = filter_alpha;
    return 0;
}

ASHMD_EXPORT int32_t ashmd_set_bonded_tables(
    void* ptr,
    int32_t n_bonds, const int32_t* bond_i, const int32_t* bond_j, const double* bond_kr, const double* bond_req,
    int32_t n_angles, const int32_t* ang_i, const int32_t* ang_j, const int32_t* ang_k, const double* ang_kt, const double* ang_eq,
    int32_t n_dih, const int32_t* dih_i, const int32_t* dih_j, const int32_t* dih_k, const int32_t* dih_l,
    const double* dih_pk, const double* dih_n, const double* dih_phase, const int32_t* dih_is_pep,
    int32_t n_p14, const int32_t* p14_i, const int32_t* p14_j, const double* p14_qprod, const double* p14_rmin, const double* p14_eps,
    const int32_t* excl_offsets, int32_t n_excl_targets, const int32_t* excl_targets
) {
    auto* h = static_cast<EngineHandle*>(ptr);
    if (!h) return -1;
    auto& t = h->top;

    t.bonds.resize(n_bonds);
    for (int32_t b = 0; b < n_bonds; ++b) {
        t.bonds[b] = {bond_i[b], bond_j[b], bond_kr[b], bond_req[b]};
    }

    t.angles.resize(n_angles);
    for (int32_t a = 0; a < n_angles; ++a) {
        t.angles[a] = {ang_i[a], ang_j[a], ang_k[a], ang_kt[a], ang_eq[a]};
    }

    t.dihedrals.resize(n_dih);
    for (int32_t d = 0; d < n_dih; ++d) {
        t.dihedrals[d] = {dih_i[d], dih_j[d], dih_k[d], dih_l[d], dih_pk[d], dih_n[d], dih_phase[d], dih_is_pep[d]};
    }

    t.pairs14.resize(n_p14);
    for (int32_t p = 0; p < n_p14; ++p) {
        t.pairs14[p] = {p14_i[p], p14_j[p], p14_qprod[p], p14_rmin[p], p14_eps[p], 0};
    }

    t.excl_offsets.assign(excl_offsets, excl_offsets + t.n_atoms + 1);
    if (n_excl_targets > 0 && excl_targets != nullptr) {
        t.excl_targets.assign(excl_targets, excl_targets + n_excl_targets);
    } else {
        t.excl_targets.clear();
    }
    return 0;
}

ASHMD_EXPORT int32_t ashmd_set_constraints_and_filter(
    void* ptr,
    int32_t n_cons, const int32_t* con_i, const int32_t* con_j, const double* con_d2,
    int32_t n_wat, const int32_t* wat_o, const int32_t* wat_h1, const int32_t* wat_h2,
    int32_t n_fg, const int32_t* fg_heavy, const int32_t* fg_nh, const int32_t* fg_h_flat
) {
    auto* h = static_cast<EngineHandle*>(ptr);
    if (!h) return -1;
    auto& t = h->top;

    t.constraints.resize(n_cons);
    for (int32_t c = 0; c < n_cons; ++c) {
        t.constraints[c] = {con_i[c], con_j[c], con_d2[c]};
    }

    t.waters.resize(n_wat);
    for (int32_t w = 0; w < n_wat; ++w) {
        t.waters[w] = {wat_o[w], wat_h1[w], wat_h2[w]};
    }

    t.filter_groups.resize(n_fg);
    for (int32_t g = 0; g < n_fg; ++g) {
        ashmd::CovalentFilterGroup grp{};
        grp.heavy_idx = fg_heavy[g];
        grp.n_hydrogens = std::min(4, std::max(0, fg_nh[g]));
        double msum = t.mass[grp.heavy_idx];
        for (int k = 0; k < grp.n_hydrogens; ++k) {
            const int32_t hidx = fg_h_flat[g * 4 + k];
            grp.h_indices[k] = hidx;
            msum += t.mass[hidx];
        }
        grp.inv_total_mass = (msum > 0.0) ? (1.0 / msum) : 0.0;
        t.filter_groups[g] = grp;
    }
    return 0;
}

ASHMD_EXPORT int32_t ashmd_set_tethers_and_cap(
    void* ptr,
    const double* ref_x, const double* ref_y, const double* ref_z,
    const double* tether_k,
    int32_t n_anchors, const int32_t* cap_anchors,
    int32_t n_shell_wat, const int32_t* shell_water_oxygens,
    double cap_radius, double cap_k, double cap_soft_beta
) {
    auto* h = static_cast<EngineHandle*>(ptr);
    if (!h) return -1;
    auto& t = h->top;
    const int32_t n = t.n_atoms;
    if (ref_x && ref_y && ref_z && tether_k) {
        t.ref_x.assign(ref_x, ref_x + n);
        t.ref_y.assign(ref_y, ref_y + n);
        t.ref_z.assign(ref_z, ref_z + n);
        t.tether_k.assign(tether_k, tether_k + n);
    }
    if (n_anchors > 0 && cap_anchors) {
        t.cap_anchors.assign(cap_anchors, cap_anchors + n_anchors);
    } else {
        t.cap_anchors.clear();
    }
    if (n_shell_wat > 0 && shell_water_oxygens) {
        t.shell_water_oxygens.assign(shell_water_oxygens, shell_water_oxygens + n_shell_wat);
    } else {
        t.shell_water_oxygens.clear();
    }
    t.cap_radius = cap_radius;
    t.cap_k = cap_k;
    t.cap_soft_beta = cap_soft_beta;
    return 0;
}

ASHMD_EXPORT int32_t ashmd_init_replicas(
    void* ptr,
    int32_t n_replicas,
    const double* lambda_inter,
    const double* mu_torsion,
    const double* softcore_alpha,
    uint64_t global_seed,
    const double* init_rx,
    const double* init_ry,
    const double* init_rz,
    double init_temp_k
) {
    auto* h = static_cast<EngineHandle*>(ptr);
    if (!h || h->top.n_atoms <= 0) return -1;

    h->setup_replicas(n_replicas, lambda_inter, mu_torsion, softcore_alpha, global_seed);
    const int32_t n = h->top.n_atoms;
    for (int32_t r = 0; r < n_replicas; ++r) {
        std::memcpy(h->replicas[r].rx.data(), init_rx, n * sizeof(double));
        std::memcpy(h->replicas[r].ry.data(), init_ry, n * sizeof(double));
        std::memcpy(h->replicas[r].rz.data(), init_rz, n * sizeof(double));
        if (init_temp_k > 0.0) {
            ashmd::initialize_velocities(h->top, h->replicas[r], global_seed, init_temp_k);
        }
    }
    return 0;
}

ASHMD_EXPORT int32_t ashmd_set_state(
    void* ptr,
    int32_t replica_idx,
    const double* rx, const double* ry, const double* rz,
    const double* vx_ang_ps, const double* vy_ang_ps, const double* vz_ang_ps
) {
    auto* h = static_cast<EngineHandle*>(ptr);
    if (!h || replica_idx < 0 || replica_idx >= static_cast<int32_t>(h->replicas.size())) return -1;
    auto& st = h->replicas[replica_idx];
    const int32_t n = h->top.n_atoms;
    if (rx && ry && rz) {
        std::memcpy(st.rx.data(), rx, n * sizeof(double));
        std::memcpy(st.ry.data(), ry, n * sizeof(double));
        std::memcpy(st.rz.data(), rz, n * sizeof(double));
        st.pme_ws.initialized = false;
        st.ash_ws.initialized = false;
        st.ash_ws.scaf_static_ready = false;
        st.forces_initialized = false;
    }
    if (vx_ang_ps && vy_ang_ps && vz_ang_ps) {
        const double inv_tf = 1.0 / ashmd::TIME_FACTOR;
        for (int32_t i = 0; i < n; ++i) {
            st.vx[i] = vx_ang_ps[i] * inv_tf;
            st.vy[i] = vy_ang_ps[i] * inv_tf;
            st.vz[i] = vz_ang_ps[i] * inv_tf;
        }
    }
    return 0;
}

ASHMD_EXPORT int32_t ashmd_get_state(
    void* ptr,
    int32_t replica_idx,
    double* rx_out, double* ry_out, double* rz_out,
    double* vx_ang_ps_out, double* vy_ang_ps_out, double* vz_ang_ps_out,
    AshmdEnergyC* energy_out
) {
    auto* h = static_cast<EngineHandle*>(ptr);
    if (!h || replica_idx < 0 || replica_idx >= static_cast<int32_t>(h->replicas.size())) return -1;
    auto& st = h->replicas[replica_idx];
    const int32_t n = h->top.n_atoms;

    if (!st.forces_initialized) {
        ashmd::evaluate_all_forces(h->top, st, *h->rep_pools[replica_idx], ashmd::PHYSIOLOGICAL_TEMP_K);
    }

    if (rx_out && ry_out && rz_out) {
        std::memcpy(rx_out, st.rx.data(), n * sizeof(double));
        std::memcpy(ry_out, st.ry.data(), n * sizeof(double));
        std::memcpy(rz_out, st.rz.data(), n * sizeof(double));
    }
    if (vx_ang_ps_out && vy_ang_ps_out && vz_ang_ps_out) {
        for (int32_t i = 0; i < n; ++i) {
            vx_ang_ps_out[i] = st.vx[i] * ashmd::TIME_FACTOR;
            vy_ang_ps_out[i] = st.vy[i] * ashmd::TIME_FACTOR;
            vz_ang_ps_out[i] = st.vz[i] * ashmd::TIME_FACTOR;
        }
    }
    if (energy_out) {
        copy_energy_to_c(st.energy_total, energy_out);
    }
    return 0;
}

ASHMD_EXPORT int32_t ashmd_eval_single_point(
    void* ptr,
    int32_t replica_idx,
    double target_temp_k,
    double* fx_out, double* fy_out, double* fz_out,
    AshmdEnergyC* energy_out
) {
    auto* h = static_cast<EngineHandle*>(ptr);
    if (!h || replica_idx < 0 || replica_idx >= static_cast<int32_t>(h->replicas.size())) return -1;
    auto& st = h->replicas[replica_idx];
    ashmd::evaluate_all_forces(h->top, st, *h->rep_pools[replica_idx], target_temp_k);
    const int32_t n = h->top.n_atoms;
    if (fx_out && fy_out && fz_out) {
        for (int32_t i = 0; i < n; ++i) {
            fx_out[i] = st.fx_fast[i] + st.fx_slow[i];
            fy_out[i] = st.fy_fast[i] + st.fy_slow[i];
            fz_out[i] = st.fz_fast[i] + st.fz_slow[i];
        }
    }
    if (energy_out) {
        copy_energy_to_c(st.energy_total, energy_out);
    }
    return 0;
}

ASHMD_EXPORT int32_t ashmd_eval_binding_dg(
    void* ptr,
    const double* rx,
    const double* ry,
    const double* rz,
    double target_temp_k,
    double sasa_gamma_gbsa,
    double eps_in_gbsa,
    double* out_terms5
) {
    auto* h = static_cast<EngineHandle*>(ptr);
    if (!h || !rx || !ry || !rz || !out_terms5) return -1;
    ashmd::compute_exact_mmgbsa_binding(
        h->top, *h->main_pool, rx, ry, rz, target_temp_k, sasa_gamma_gbsa, eps_in_gbsa, out_terms5
    );
    return 0;
}

ASHMD_EXPORT int32_t ashmd_eval_binding_rd_gbsa_sasa(
    void* ptr,
    const double* rx,
    const double* ry,
    const double* rz,
    double target_temp_k,
    double cav_gamma,
    double disp_scale,
    double* out_terms8,
    double* sasa_free_out,
    double* sasa_bound_out
) {
    auto* h = static_cast<EngineHandle*>(ptr);
    if (!h || !rx || !ry || !rz || !out_terms8) return -1;
    ashmd::compute_rd_gbsa_wca_binding(
        h->top, *h->main_pool, rx, ry, rz, target_temp_k, cav_gamma, disp_scale,
        out_terms8, sasa_free_out, sasa_bound_out
    );
    return 0;
}

ASHMD_EXPORT int32_t ashmd_set_replica_tempering(
    void* ptr,
    const double* lambda_inter,
    const double* mu_torsion,
    const double* softcore_alpha,
    double eval_temp_k
) {
    auto* h = static_cast<EngineHandle*>(ptr);
    if (!h || h->replicas.empty() || !lambda_inter || !mu_torsion || !softcore_alpha) return -1;
    const int32_t n_rep = static_cast<int32_t>(h->replicas.size());
    const double t_k = (eval_temp_k > 0.0) ? eval_temp_k : ashmd::PHYSIOLOGICAL_TEMP_K;
    for (int32_t r = 0; r < n_rep; ++r) {
        h->replicas[r].tempering.lambda_inter   = lambda_inter[r];
        h->replicas[r].tempering.mu_torsion     = mu_torsion[r];
        h->replicas[r].tempering.softcore_alpha = softcore_alpha[r];
        h->replicas[r].forces_initialized       = false;
    }
    h->main_pool->parallel_ntasks(n_rep, [&](int32_t /*tid*/, int32_t r) {
        ashmd::evaluate_all_forces(h->top, h->replicas[r], *h->rep_pools[r], t_k);
    });
    return 0;
}

ASHMD_EXPORT int32_t ashmd_eval_cross_potentials(
    void* ptr,
    double target_temp_k,
    double* u_cross_flat_out
) {
    auto* h = static_cast<EngineHandle*>(ptr);
    if (!h || h->replicas.empty() || !u_cross_flat_out) return -1;
    std::vector<std::vector<double>> u_cross;
    ashmd::compute_cross_reduced_potentials(
        h->top, h->replicas, *h->serial_pool, target_temp_k, u_cross
    );
    const int32_t M = static_cast<int32_t>(h->replicas.size());
    for (int32_t r = 0; r < M; ++r) {
        std::memcpy(u_cross_flat_out + static_cast<size_t>(r) * M, u_cross[r].data(), M * sizeof(double));
    }
    return 0;
}

ASHMD_EXPORT int32_t ashmd_minimize(
    void* ptr,
    int32_t max_steps,
    double fmax_tol,
    double init_temp_after_k
) {
    auto* h = static_cast<EngineHandle*>(ptr);
    if (!h || h->replicas.empty()) return -1;
    const int32_t n_rep = static_cast<int32_t>(h->replicas.size());

    // Minimize Replica 0 on the physical steric potential (using full main_pool threads),
    // then broadcast the clash-relaxed structure to all replicas 1..M-1 before independent MB velocity draw.
    const auto saved_tp0 = h->replicas[0].tempering;
    h->replicas[0].tempering = {1.0, 1.0, 0.0};
    const int32_t steps0 = ashmd::minimize_energy_sd(h->top, h->replicas[0], *h->main_pool, max_steps, fmax_tol);
    h->replicas[0].tempering = saved_tp0;

    for (int32_t r = 1; r < n_rep; ++r) {
        h->replicas[r].rx = h->replicas[0].rx;
        h->replicas[r].ry = h->replicas[0].ry;
        h->replicas[r].rz = h->replicas[0].rz;
        h->replicas[r].pme_ws.initialized = false;
        h->replicas[r].ash_ws.initialized = false;
        h->replicas[r].forces_initialized = false;
    }

    h->main_pool->parallel_ntasks(n_rep, [&](int32_t /*tid*/, int32_t r) {
        const double eval_t = (init_temp_after_k > 0.0) ? init_temp_after_k : ashmd::PHYSIOLOGICAL_TEMP_K;
        ashmd::evaluate_all_forces(h->top, h->replicas[r], *h->rep_pools[r], eval_t);
        if (init_temp_after_k > 0.0) {
            ashmd::initialize_velocities(h->top, h->replicas[r], h->global_seed, init_temp_after_k);
            ashmd::combine_energies(h->top, h->replicas[r]);
        }
    });
    return steps0;
}

// Run n_outer_steps of FC-BAOAB dynamics across all replicas (with optional IHRT swaps & MBAR recording)
ASHMD_EXPORT int32_t ashmd_run_steps(
    void* ptr,
    int32_t n_outer_steps,
    double dt_ps,
    int32_t mts_ratio,
    double gamma_ps_inv,
    double target_temp_k,
    int32_t exchange_every_outer,
    int32_t record_mbar
) {
    auto* h = static_cast<EngineHandle*>(ptr);
    if (!h || h->replicas.empty() || n_outer_steps < 0) return -1;

    const int32_t n_rep = static_cast<int32_t>(h->replicas.size());

    for (int32_t step = 0; step < n_outer_steps; ++step) {
        if (n_rep == 1) {
            ashmd::step_fc_baoab_outer(
                h->top, h->replicas[0], *h->rep_pools[0],
                h->global_seed, h->inner_step_counters[0],
                dt_ps, mts_ratio, gamma_ps_inv, target_temp_k
            );
        } else {
            h->main_pool->parallel_ntasks(n_rep, [&](int32_t /*tid*/, int32_t r) {
                ashmd::step_fc_baoab_outer(
                    h->top, h->replicas[r], *h->rep_pools[r],
                    h->global_seed, h->inner_step_counters[r],
                    dt_ps, mts_ratio, gamma_ps_inv, target_temp_k
                );
            });
        }
        h->outer_step_done += 1ULL;

        if (n_rep > 1 && exchange_every_outer > 0 && (h->outer_step_done % static_cast<uint64_t>(exchange_every_outer) == 0ULL)) {
            ashmd::attempt_ihrt_swaps(
                h->top, h->replicas, *h->serial_pool,
                h->global_seed, h->exchange_round, target_temp_k,
                h->swap_attempts, h->swap_accepts,
                record_mbar ? &h->mbar_acc : nullptr
            );
            h->exchange_round += 1ULL;
        }
    }
    return 0;
}

ASHMD_EXPORT int32_t ashmd_get_mbar_summary(
    void* ptr,
    int32_t* n_samples_out,
    double* f_states_out,
    double* w0_weights_out,
    int32_t* origin_states_out,
    int64_t* swap_attempts_out,
    int64_t* swap_accepts_out
) {
    auto* h = static_cast<EngineHandle*>(ptr);
    if (!h) return -1;
    const int32_t N = h->mbar_acc.total_samples();
    if (n_samples_out) *n_samples_out = N;

    if (f_states_out && w0_weights_out && N > 0) {
        std::vector<double> f_vec, w0_vec;
        h->mbar_acc.solve_mbar(f_vec, w0_vec);
        std::memcpy(f_states_out, f_vec.data(), f_vec.size() * sizeof(double));
        std::memcpy(w0_weights_out, w0_vec.data(), w0_vec.size() * sizeof(double));
        if (origin_states_out) {
            std::memcpy(origin_states_out, h->mbar_acc.origin_state.data(), N * sizeof(int32_t));
        }
    }
    if (swap_attempts_out && !h->swap_attempts.empty()) {
        std::memcpy(swap_attempts_out, h->swap_attempts.data(), h->swap_attempts.size() * sizeof(int64_t));
    }
    if (swap_accepts_out && !h->swap_accepts.empty()) {
        std::memcpy(swap_accepts_out, h->swap_accepts.data(), h->swap_accepts.size() * sizeof(int64_t));
    }
    return 0;
}

ASHMD_EXPORT int32_t ashmd_save_checkpoint(void* ptr, const char* path) {
    auto* h = static_cast<EngineHandle*>(ptr);
    if (!h || !path) return -1;
    std::string err;
    const bool ok = ashmd::save_checkpoint_atomic(
        path, h->top, h->replicas, h->inner_step_counters,
        h->global_seed, h->outer_step_done, h->exchange_round,
        h->swap_attempts, h->swap_accepts, h->mbar_acc, err
    );
    if (!ok) {
        h->last_error = err;
        return -2;
    }
    return 0;
}

ASHMD_EXPORT int32_t ashmd_load_checkpoint(void* ptr, const char* path, uint64_t* outer_step_done_out) {
    auto* h = static_cast<EngineHandle*>(ptr);
    if (!h || !path) return -1;
    std::string err;
    const bool ok = ashmd::load_checkpoint_verified(
        path, h->top, h->replicas, h->inner_step_counters,
        h->global_seed, h->outer_step_done, h->exchange_round,
        h->swap_attempts, h->swap_accepts, h->mbar_acc, err
    );
    if (!ok) {
        h->last_error = err;
        return -2;
    }
    const int32_t n_rep = static_cast<int32_t>(h->replicas.size());
    h->rep_pools.clear();
    const int32_t inner_threads = (n_rep == 1) ? h->main_pool->num_threads() : 1;
    for (int32_t r = 0; r < n_rep; ++r) {
        h->rep_pools.push_back(std::make_unique<ashmd::ParallelPool>(inner_threads));
    }
    if (outer_step_done_out) *outer_step_done_out = h->outer_step_done;
    return 0;
}
