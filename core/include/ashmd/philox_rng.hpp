#pragma once

// Counter-based deterministic Philox-4x32-10 PRNG (Salmon et al., SC 2011).
// Key property for ASH-MD:
//   Random numbers for the Langevin 'O' thermostat step are pure mathematical functions
//   of (global_seed, replica_id, step_index, atom_index).
//   Therefore:
//     1) 1 thread vs 4 threads vs 10 threads yield 100% bit-identical thermal kicks.
//     2) Stopping at step K and resuming from checkpoint at step K yields 100% bit-identical
//        trajectories without storing megabytes of PRNG state tables.

#include <cmath>
#include <cstdint>
#include <array>

namespace ashmd {

class Philox4x32 {
public:
    static constexpr uint32_t PHILOX_M0 = 0xD2511F53U;
    static constexpr uint32_t PHILOX_M1 = 0xCD9E8D57U;
    static constexpr uint32_t PHILOX_W0 = 0x9E3779B9U;
    static constexpr uint32_t PHILOX_W1 = 0xBB67AE85U;

    static inline std::array<uint32_t, 4> generate(
        uint64_t seed,
        uint32_t stream_id,
        uint64_t step,
        uint32_t atom_idx
    ) noexcept {
        uint32_t c0 = static_cast<uint32_t>(step & 0xFFFFFFFFULL);
        uint32_t c1 = static_cast<uint32_t>(step >> 32);
        uint32_t c2 = atom_idx;
        uint32_t c3 = stream_id;

        uint32_t k0 = static_cast<uint32_t>(seed & 0xFFFFFFFFULL);
        uint32_t k1 = static_cast<uint32_t>(seed >> 32);

        for (int round = 0; round < 10; ++round) {
            uint64_t p0 = static_cast<uint64_t>(PHILOX_M0) * static_cast<uint64_t>(c0);
            uint64_t p1 = static_cast<uint64_t>(PHILOX_M1) * static_cast<uint64_t>(c2);

            uint32_t hi0 = static_cast<uint32_t>(p0 >> 32);
            uint32_t lo0 = static_cast<uint32_t>(p0 & 0xFFFFFFFFULL);
            uint32_t hi1 = static_cast<uint32_t>(p1 >> 32);
            uint32_t lo1 = static_cast<uint32_t>(p1 & 0xFFFFFFFFULL);

            c0 = hi1 ^ c1 ^ k0;
            c1 = lo1;
            c2 = hi0 ^ c3 ^ k1;
            c3 = lo0;

            k0 += PHILOX_W0;
            k1 += PHILOX_W1;
        }
        return {c0, c1, c2, c3};
    }

    // Generate 3 independent standard normal N(0, 1) variates for atom (gx, gy, gz)
    // using Box-Muller with strict zero-avoidance mapping u in (0, 1].
    static inline void normal3(
        uint64_t seed,
        uint32_t replica_id,
        uint64_t step,
        uint32_t atom_idx,
        double& gx,
        double& gy,
        double& gz
    ) noexcept {
        auto r = generate(seed, replica_id, step, atom_idx);
        // Map uint32 in [0, 2^32-1] -> (0, 1] strictly away from 0
        constexpr double INV_2P32 = 1.0 / 4294967296.0;
        constexpr double TWO_PI   = 6.28318530717958647692;

        double u1 = (static_cast<double>(r[0]) + 1.0) * INV_2P32;
        double u2 = (static_cast<double>(r[1]) + 0.5) * INV_2P32;
        double u3 = (static_cast<double>(r[2]) + 1.0) * INV_2P32;
        double u4 = (static_cast<double>(r[3]) + 0.5) * INV_2P32;

        double rad1 = std::sqrt(-2.0 * std::log(u1));
        double rad2 = std::sqrt(-2.0 * std::log(u3));

        gx = rad1 * std::cos(TWO_PI * u2);
        gy = rad1 * std::sin(TWO_PI * u2);
        gz = rad2 * std::cos(TWO_PI * u4);
    }

    // Uniform (0, 1) variate for Metropolis-Hastings replica exchange decisions
    static inline double uniform01(
        uint64_t seed,
        uint64_t step,
        uint32_t pair_idx
    ) noexcept {
        auto r = generate(seed, 0x9E370001U, step, pair_idx);
        constexpr double INV_2P32 = 1.0 / 4294967296.0;
        return (static_cast<double>(r[0]) + 0.5) * INV_2P32;
    }
};

} // namespace ashmd
