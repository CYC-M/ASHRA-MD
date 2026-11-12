#pragma once

#include <cmath>
#include <cstdint>
#include <vector>
#include <array>
#include <algorithm>

namespace ashmd {

// Physical constants in ASH-MD internal units:
//   Length:      Angstrom (A)
//   Energy:      kcal/mol
//   Mass:        amu (g/mol)
//   Charge:      elementary charge (e)
//   Temperature: Kelvin (K)
//   Internal time tau = t_ps * TIME_FACTOR, so K = 0.5 * m * v_int^2 [kcal/mol]
constexpr double BOLTZMANN_KCAL = 0.00198720425864083;    // kcal/(mol*K)
constexpr double COULOMB_KCAL_A = 332.05221729;           // AMBER (18.2223)^2 kcal*A/(mol*e^2)
constexpr double TIME_FACTOR    = 20.45482828087295;      // sqrt(418.4)
constexpr double PI_VAL         = 3.14159265358979323846;
constexpr double PHYSIOLOGICAL_TEMP_K = 310.15;           // 37.00 C

// Deterministic Q31.32 fixed-point force accumulation scale.
// Associative & commutative int64_t addition guarantees bit-exact force sums
// regardless of thread count (1 vs 4 vs N threads) or dynamic work scheduling.
constexpr double FORCE_FIXED_SCALE     = 4294967296.0;    // 2^32
constexpr double INV_FORCE_FIXED_SCALE = 1.0 / 4294967296.0;
constexpr double FORCE_CLAMP_MAX       = 1.0e7;           // kcal/(mol*A); >214x headroom under INT64_MAX

inline int64_t double_to_qforce(double f) noexcept {
    if (f > FORCE_CLAMP_MAX) f = FORCE_CLAMP_MAX;
    if (f < -FORCE_CLAMP_MAX) f = -FORCE_CLAMP_MAX;
    return static_cast<int64_t>(std::llround(f * FORCE_FIXED_SCALE));
}

inline double qforce_to_double(int64_t qf) noexcept {
    return static_cast<double>(qf) * INV_FORCE_FIXED_SCALE;
}

// Deterministic Q87.40 128-bit fixed-point energy & Born-derivative accumulation:
//   - Fractional scale 2^40 (~9.09e-13 kcal/mol resolution -> <3e-6 finite-difference noise at 1e-5 A)
//   - 128-bit signed integer Abelian group (__int128 on 64-bit Clang/GCC) provides >1.5e20 clashed-pair
//     headroom so steric overlaps can NEVER wrap around or lose cross-thread associative exactness!
#if defined(__SIZEOF_INT128__)
__extension__ using qenergy_t = __int128;
#else
using qenergy_t = int64_t;
#endif

constexpr double ENERGY_FIXED_SCALE     = 1099511627776.0;    // 2^40
constexpr double INV_ENERGY_FIXED_SCALE = 1.0 / 1099511627776.0;
constexpr double ENERGY_CLAMP_MAX       = 1.0e6;              // per-pair clamp [kcal/mol]

inline qenergy_t double_to_qenergy(double e) noexcept {
    if (e > ENERGY_CLAMP_MAX) e = ENERGY_CLAMP_MAX;
    if (e < -ENERGY_CLAMP_MAX) e = -ENERGY_CLAMP_MAX;
    return static_cast<qenergy_t>(std::llround(e * ENERGY_FIXED_SCALE));
}

inline double qenergy_to_double(qenergy_t qe) noexcept {
    return static_cast<double>(qe) * INV_ENERGY_FIXED_SCALE;
}

// Deterministic Q19.44 fixed-point OBC descreening integral accumulation (resolution ~5.7e-14)
constexpr double DESCREEN_FIXED_SCALE     = 17592186044416.0; // 2^44
constexpr double INV_DESCREEN_FIXED_SCALE = 1.0 / 17592186044416.0;
constexpr double DESCREEN_CLAMP_MAX       = 1.0e4;

inline int64_t double_to_qdescreen(double v) noexcept {
    if (v > DESCREEN_CLAMP_MAX) v = DESCREEN_CLAMP_MAX;
    if (v < -DESCREEN_CLAMP_MAX) v = -DESCREEN_CLAMP_MAX;
    return static_cast<int64_t>(std::llround(v * DESCREEN_FIXED_SCALE));
}

inline double qdescreen_to_double(int64_t qv) noexcept {
    return static_cast<double>(qv) * INV_DESCREEN_FIXED_SCALE;
}

// Atom zone classification for Active-Shell Hamiltonian (ASH)
enum AtomZone : int32_t {
    ZONE_ACTIVE_SOLUTE = 0,  // Zone A: Pocket + Peptide + Structural Metal Ions (full flex)
    ZONE_ACTIVE_WATER  = 1,  // Zone A: Explicit interfacial hydration shell waters
    ZONE_BUFFER_RIM    = 2,  // Zone B: Allosteric rim with smooth harmonic backbone tether
    ZONE_SCAFFOLD      = 3   // Zone C: Distant receptor scaffold (anchored/quasi-rigid field)
};

// Atom chain/role tag for Interface Hamiltonian Replica Tempering (IHRT)
enum SoluteRole : int32_t {
    ROLE_RECEPTOR = 0,
    ROLE_PEPTIDE  = 1,
    ROLE_ION      = 2,
    ROLE_SOLVENT  = 3
};

struct BondTerm {
    int32_t i, j;
    double kr;   // AMBER convention: U = kr * (r - req)^2 [kcal/(mol*A^2)]
    double req;  // Equilibrium bond length [A]
};

struct AngleTerm {
    int32_t i, j, k;
    double ktheta;   // AMBER convention: U = ktheta * (theta - theta_eq)^2 [kcal/(mol*rad^2)]
    double theta_eq; // Equilibrium angle [rad]
};

struct DihedralTerm {
    int32_t i, j, k, l;
    double pk;       // Barrier half-height V_n / idivf [kcal/mol]
    double periodicity; // n >= 1
    double phase;    // gamma [rad]
    int32_t is_peptide_torsion; // 1 if scaled by mu_m in IHRT replica tempering
};

struct Pair14Term {
    int32_t i, j;
    double q_prod_scaled; // C_coul * q_i * q_j / scee (scee = 1.2)
    double rmin_ij;       // rmin_half_i + rmin_half_j [A]
    double eps_scaled;    // sqrt(eps_i * eps_j) / scnb (scnb = 2.0)
    int32_t is_interface; // Always 0 for intramolecular 1-4, kept for symmetry
};

struct HolonomicConstraint {
    int32_t i, j;
    double dist_sq; // Target squared bond distance d_0^2 [A^2]
};

struct WaterTriplet {
    int32_t o, h1, h2; // Rigid TIP3P water indices for analytical SETTLE
};

// Linear Topological Covalent-Centroid Filter entry for FC-BAOAB resonance suppression:
// q_tilde_i = q_i + alpha_i * (q_heavy_parent - q_i) = (1 - alpha_i)*q_i + alpha_i*q_parent
// Wait: for exact momentum-preserving mass-weighted group centroid:
// Group g = {heavy parent p, attached hydrogens h_1..h_k} with total mass M_g:
// q_com(g) = (m_p * q_p + sum_a m_a * q_a) / M_g
// Filtered position: q_tilde_x = (1 - filter_alpha) * q_x + filter_alpha * q_com(g)
// Since this is a constant linear map P independent of q, U_slow(P q) is 100% conservative!
struct CovalentFilterGroup {
    int32_t heavy_idx;
    int32_t n_hydrogens;
    int32_t h_indices[4];
    double inv_total_mass;
};

struct EnergyBreakdown {
    double bond       = 0.0;
    double angle      = 0.0;
    double dihedral   = 0.0;
    double pair14_lj  = 0.0;
    double pair14_el  = 0.0;
    double lj_sr      = 0.0;
    double lj_disp    = 0.0;
    double elec_dir   = 0.0;
    double elec_recip = 0.0;
    double elec_self  = 0.0;
    double elec_excl  = 0.0;
    double gb_pol     = 0.0;
    double np_sasa    = 0.0;
    double cap_shell  = 0.0;
    double tether_buf = 0.0;
    double potential  = 0.0;
    double kinetic    = 0.0;
    double total      = 0.0;
    double temperature = 0.0;

    // Unscaled physical components needed for fast exact IHRT replica cross-energy evaluation
    double raw_pep_torsion     = 0.0;
    double raw_interface_nonb  = 0.0;

    void finalize_potential() noexcept {
        potential = bond + angle + dihedral + pair14_lj + pair14_el
                  + lj_sr + lj_disp + elec_dir + elec_recip + elec_self + elec_excl
                  + gb_pol + np_sasa + cap_shell + tether_buf;
        total = potential + kinetic;
    }
};

enum SolventMode : int32_t {
    MODE_EXPLICIT_PME = 0, // Periodic box + TIP3P + 4th-order B-Spline SPME
    MODE_ACTIVE_SHELL = 1  // Explicit hydration shell + OBC2 GB + ACE NP + Softmin Cap
};

struct SystemTopology {
    int32_t n_atoms = 0;
    SolventMode solvent_mode = MODE_ACTIVE_SHELL;

    std::vector<double> mass;          // [amu]
    std::vector<double> inv_mass;      // 1/mass (0.0 for frozen scaffold atoms)
    std::vector<double> charge;        // [e]
    std::vector<double> rmin_half;     // AMBER Rmin/2 [A]
    std::vector<double> epsilon;       // AMBER LJ well depth [kcal/mol]
    std::vector<double> gb_radius;     // AMBER mbondi2 intrinsic Born radius [A]
    std::vector<double> gb_screen;     // OBC2 descreening factor S_i
    std::vector<double> res_diel;      // Residue-specific interior dielectric eps_i (RD-OBC2; default 2.0)

    std::vector<int32_t> zone;         // AtomZone
    std::vector<int32_t> role;         // SoluteRole

    // Reference coordinates & per-atom harmonic restraint constants (for Zone B rim or NVT eq)
    std::vector<double> ref_x, ref_y, ref_z;
    std::vector<double> tether_k;      // U_tether = 0.5 * k_i * |r_i - r0_i|^2

    // Anchor atom indices defining the moving pocket+peptide envelope for Softmin Hydration Cap
    std::vector<int32_t> cap_anchors;
    std::vector<int32_t> shell_water_oxygens;
    double cap_radius    = 6.5;        // Max distance from interface envelope before soft wall [A]
    double cap_k         = 3.0;        // Cap harmonic force constant [kcal/(mol*A^2)]
    double cap_soft_beta = 1.5;        // Log-sum-exp inverse smoothing length [1/A]

    // Bonded & constraint tables
    std::vector<BondTerm> bonds;
    std::vector<AngleTerm> angles;
    std::vector<DihedralTerm> dihedrals;
    std::vector<Pair14Term> pairs14;
    std::vector<HolonomicConstraint> constraints; // X-H RATTLE constraints
    std::vector<WaterTriplet> waters;             // Rigid TIP3P waters for SETTLE
    std::vector<CovalentFilterGroup> filter_groups; // Topological groups for FC-BAOAB

    // CSR-compressed exclusion list (1-2, 1-3, and 1-4 bonded neighbors, sorted per atom i < j)
    std::vector<int32_t> excl_offsets; // length n_atoms + 1
    std::vector<int32_t> excl_targets; // sorted neighbor indices j > i

    // Periodic box & nonbonded parameters (Mode 0: PME)
    double box_x = 0.0, box_y = 0.0, box_z = 0.0;
    double cutoff     = 9.0;           // Short-range cutoff [A]
    double switch_on  = 7.5;           // LJ C2 force/potential switch start [A]
    double ewald_alpha = 0.348;        // Ewald splitting alpha [1/A]
    int32_t pme_grid_x = 32, pme_grid_y = 32, pme_grid_z = 32;

    // Continuum / Active-Shell parameters (Mode 1: ASH)
    double gb_cutoff      = 16.0;      // Cutoff for GB & nonbonded in ASH mode [A]
    double eps_in         = 1.0;
    double eps_out        = 78.5;
    double ionic_strength = 0.15;      // Physiological 0.15 M NaCl
    double sasa_gamma     = 0.0054;    // Nonpolar surface tension [kcal/(mol*A^2)]

    // FC-BAOAB filter strength (0.0 = unfiltered standard BAOAB, 0.35 = optimal X-H resonance filter)
    double filter_alpha = 0.35;

    bool is_excluded(int32_t i, int32_t j) const noexcept {
        if (i == j) return true;
        if (i > j) std::swap(i, j);
        const int32_t start = excl_offsets[i];
        const int32_t end   = excl_offsets[i + 1];
        return std::binary_search(excl_targets.begin() + start, excl_targets.begin() + end, j);
    }

    int32_t degrees_of_freedom() const noexcept {
        int32_t active_atoms = 0;
        for (int32_t i = 0; i < n_atoms; ++i) {
            if (inv_mass[i] > 0.0) ++active_atoms;
        }
        int32_t dof = 3 * active_atoms;
        // Subtract holonomic constraints: 3 internal DOFs per rigid SETTLE water + 1 per X-H RATTLE
        dof -= 3 * static_cast<int32_t>(waters.size());
        dof -= static_cast<int32_t>(constraints.size());
        // Subtract 3 COM translation DOFs if Periodic PME without external tethers
        bool has_frozen_or_tether = false;
        for (int32_t i = 0; i < n_atoms; ++i) {
            if (inv_mass[i] == 0.0 || tether_k[i] > 0.0) {
                has_frozen_or_tether = true;
                break;
            }
        }
        if (!has_frozen_or_tether && dof > 6) {
            dof -= 3;
        }
        return std::max(1, dof);
    }
};

// Hamiltonian Tempering scaling parameters for a single replica m in IHRT
struct ReplicaTemperingParams {
    double lambda_inter = 1.0; // Scales peptide-receptor / peptide-water nonbonded energy
    double mu_torsion   = 1.0; // Scales peptide rotatable torsions (typically sqrt(lambda))
    double softcore_alpha = 0.0; // Steric soft-core smoothing alpha_max * (1 - lambda)
};

} // namespace ashmd
