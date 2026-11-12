#pragma once

// Cryptographically-Verified Atomic Checkpoint Capsule (.chk) for Ultra-Long CPU Runs
//   - Designed for Instabox 8-hour node cooldowns & arbitrary multi-day CPU simulations:
//     * Self-contained FIPS 180-4 SHA-256 implementation (zero OpenSSL dependency)
//     * Atomic write-to-.tmp + POSIX rename() prevents half-written files on SIGKILL/cooldown
//     * Stores IEEE-754 exact coordinates, velocities, inner_step_counter, exchange_round,
//       swap statistics, and MBAR reduced-potential history
//     * Fail-closed verification: magic mismatch, atom count mismatch, topology hash mismatch,
//       or 1-bit SHA-256 corruption immediately aborts load with explicit error!

#include "replica_mbar.hpp"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace ashmd {

class Sha256 {
public:
    Sha256() { reset(); }

    void reset() noexcept {
        state_[0] = 0x6a09e667U; state_[1] = 0xbb67ae85U;
        state_[2] = 0x3c6ef372U; state_[3] = 0xa54ff53aU;
        state_[4] = 0x510e527fU; state_[5] = 0x9b05688cU;
        state_[6] = 0x1f83d9abU; state_[7] = 0x5be0cd19U;
        bitlen_ = 0;
        buflen_ = 0;
    }

    void update(const void* raw, size_t len) noexcept {
        const auto* data = static_cast<const uint8_t*>(raw);
        for (size_t i = 0; i < len; ++i) {
            buf_[buflen_++] = data[i];
            if (buflen_ == 64) {
                transform(buf_);
                bitlen_ += 512;
                buflen_ = 0;
            }
        }
    }

    std::array<uint8_t, 32> finalize() noexcept {
        uint32_t i = buflen_;
        if (buflen_ < 56) {
            buf_[i++] = 0x80;
            while (i < 56) buf_[i++] = 0x00;
        } else {
            buf_[i++] = 0x80;
            while (i < 64) buf_[i++] = 0x00;
            transform(buf_);
            std::memset(buf_, 0, 56);
        }
        bitlen_ += static_cast<uint64_t>(buflen_) * 8ULL;
        for (int b = 0; b < 8; ++b) {
            buf_[63 - b] = static_cast<uint8_t>((bitlen_ >> (b * 8)) & 0xFFU);
        }
        transform(buf_);

        std::array<uint8_t, 32> hash{};
        for (int j = 0; j < 8; ++j) {
            hash[j * 4 + 0] = static_cast<uint8_t>((state_[j] >> 24) & 0xFFU);
            hash[j * 4 + 1] = static_cast<uint8_t>((state_[j] >> 16) & 0xFFU);
            hash[j * 4 + 2] = static_cast<uint8_t>((state_[j] >> 8)  & 0xFFU);
            hash[j * 4 + 3] = static_cast<uint8_t>((state_[j] >> 0)  & 0xFFU);
        }
        return hash;
    }

private:
    static inline uint32_t rotr(uint32_t x, uint32_t n) noexcept {
        return (x >> n) | (x << (32 - n));
    }

    void transform(const uint8_t data[64]) noexcept {
        static constexpr uint32_t K[64] = {
            0x428a2f98U,0x71374491U,0xb5c0fbcfU,0xe9b5dba5U,0x3956c25bU,0x59f111f1U,0x923f82a4U,0xab1c5ed5U,
            0xd807aa98U,0x12835b01U,0x243185beU,0x550c7dc3U,0x72be5d74U,0x80deb1feU,0x9bdc06a7U,0xc19bf174U,
            0xe49b69c1U,0xefbe4786U,0x0fc19dc6U,0x240ca1ccU,0x2de92c6fU,0x4a7484aaU,0x5cb0a9dcU,0x76f988daU,
            0x983e5152U,0xa831c66dU,0xb00327c8U,0xbf597fc7U,0xc6e00bf3U,0xd5a79147U,0x06ca6351U,0x14292967U,
            0x27b70a85U,0x2e1b2138U,0x4d2c6dfcU,0x53380d13U,0x650a7354U,0x766a0abbU,0x81c2c92eU,0x92722c85U,
            0xa2bfe8a1U,0xa81a664bU,0xc24b8b70U,0xc76c51a3U,0xd192e819U,0xd6990624U,0xf40e3585U,0x106aa070U,
            0x19a4c116U,0x1e376c08U,0x2748774cU,0x34b0bcb5U,0x391c0cb3U,0x4ed8aa4aU,0x5b9cca4fU,0x682e6ff3U,
            0x748f82eeU,0x78a5636fU,0x84c87814U,0x8cc70208U,0x90befffaU,0xa4506cebU,0xbef9a3f7U,0xc67178f2U
        };
        uint32_t m[64];
        for (int i = 0; i < 16; ++i) {
            m[i] = (static_cast<uint32_t>(data[i * 4 + 0]) << 24)
                 | (static_cast<uint32_t>(data[i * 4 + 1]) << 16)
                 | (static_cast<uint32_t>(data[i * 4 + 2]) << 8)
                 | (static_cast<uint32_t>(data[i * 4 + 3]));
        }
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(m[i - 15], 7) ^ rotr(m[i - 15], 18) ^ (m[i - 15] >> 3);
            const uint32_t s1 = rotr(m[i - 2], 17) ^ rotr(m[i - 2], 19)  ^ (m[i - 2] >> 10);
            m[i] = m[i - 16] + s0 + m[i - 7] + s1;
        }
        uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
        uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const uint32_t ch = (e & f) ^ ((~e) & g);
            const uint32_t temp1 = h + S1 + ch + K[i] + m[i];
            const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t temp2 = S0 + maj;
            h = g; g = f; f = e; e = d + temp1;
            d = c; c = b; b = a; a = temp1 + temp2;
        }
        state_[0] += a; state_[1] += b; state_[2] += c; state_[3] += d;
        state_[4] += e; state_[5] += f; state_[6] += g; state_[7] += h;
    }

    uint32_t state_[8]{};
    uint64_t bitlen_ = 0;
    uint32_t buflen_ = 0;
    uint8_t buf_[64]{};
};

inline uint64_t compute_topology_fingerprint(const SystemTopology& top) noexcept {
    Sha256 sha;
    sha.update(&top.n_atoms, sizeof(top.n_atoms));
    int32_t sm = static_cast<int32_t>(top.solvent_mode);
    sha.update(&sm, sizeof(sm));
    if (!top.mass.empty()) sha.update(top.mass.data(), top.mass.size() * sizeof(double));
    if (!top.charge.empty()) sha.update(top.charge.data(), top.charge.size() * sizeof(double));
    int32_t nb = static_cast<int32_t>(top.bonds.size());
    int32_t nc = static_cast<int32_t>(top.constraints.size());
    sha.update(&nb, sizeof(nb));
    sha.update(&nc, sizeof(nc));
    auto digest = sha.finalize();
    uint64_t fp = 0;
    std::memcpy(&fp, digest.data(), sizeof(uint64_t));
    return fp;
}

inline bool save_checkpoint_atomic(
    const std::string& path,
    const SystemTopology& top,
    const std::vector<ReplicaState>& replicas,
    const std::vector<uint64_t>& inner_step_counters,
    uint64_t global_seed,
    uint64_t outer_step_done,
    uint64_t exchange_round,
    const std::vector<int64_t>& swap_attempts,
    const std::vector<int64_t>& swap_accepts,
    const MBARAccumulator& mbar_acc,
    std::string& err_msg
) {
    std::vector<uint8_t> payload;
    payload.reserve(4096 + static_cast<size_t>(top.n_atoms) * replicas.size() * 48);

    auto append_bytes = [&](const void* ptr, size_t sz) {
        const auto* p = static_cast<const uint8_t*>(ptr);
        payload.insert(payload.end(), p, p + sz);
    };

    const char magic[8] = {'A','S','H','M','D','C','H','K'};
    const uint32_t version = 1U;
    const int32_t n_atoms = top.n_atoms;
    const int32_t n_rep   = static_cast<int32_t>(replicas.size());
    const uint64_t top_fp = compute_topology_fingerprint(top);

    append_bytes(magic, 8);
    append_bytes(&version, sizeof(version));
    append_bytes(&n_atoms, sizeof(n_atoms));
    append_bytes(&n_rep, sizeof(n_rep));
    append_bytes(&top_fp, sizeof(top_fp));
    append_bytes(&global_seed, sizeof(global_seed));
    append_bytes(&outer_step_done, sizeof(outer_step_done));
    append_bytes(&exchange_round, sizeof(exchange_round));

    for (int32_t r = 0; r < n_rep; ++r) {
        const uint64_t isc = inner_step_counters[r];
        append_bytes(&isc, sizeof(isc));
        append_bytes(&replicas[r].tempering, sizeof(ReplicaTemperingParams));
        append_bytes(replicas[r].rx.data(), n_atoms * sizeof(double));
        append_bytes(replicas[r].ry.data(), n_atoms * sizeof(double));
        append_bytes(replicas[r].rz.data(), n_atoms * sizeof(double));
        append_bytes(replicas[r].vx.data(), n_atoms * sizeof(double));
        append_bytes(replicas[r].vy.data(), n_atoms * sizeof(double));
        append_bytes(replicas[r].vz.data(), n_atoms * sizeof(double));
    }

    const int32_t n_sw = static_cast<int32_t>(swap_attempts.size());
    append_bytes(&n_sw, sizeof(n_sw));
    if (n_sw > 0) {
        append_bytes(swap_attempts.data(), n_sw * sizeof(int64_t));
        append_bytes(swap_accepts.data(), n_sw * sizeof(int64_t));
    }

    const int32_t mbar_M = mbar_acc.n_states;
    const int32_t mbar_N = mbar_acc.total_samples();
    append_bytes(&mbar_M, sizeof(mbar_M));
    append_bytes(&mbar_N, sizeof(mbar_N));
    if (mbar_M > 0) {
        append_bytes(mbar_acc.state_counts.data(), mbar_M * sizeof(int32_t));
    }
    if (mbar_N > 0) {
        append_bytes(mbar_acc.origin_state.data(), mbar_N * sizeof(int32_t));
        append_bytes(mbar_acc.u_matrix.data(), static_cast<size_t>(mbar_N) * mbar_M * sizeof(double));
    }

    Sha256 sha;
    sha.update(payload.data(), payload.size());
    const auto digest = sha.finalize();

    const std::string tmp_path = path + ".tmp";
    {
        std::ofstream ofs(tmp_path, std::ios::binary | std::ios::trunc);
        if (!ofs) {
            err_msg = "Failed to open temporary checkpoint file for writing: " + tmp_path;
            return false;
        }
        ofs.write(reinterpret_cast<const char*>(payload.data()), static_cast<std::streamsize>(payload.size()));
        ofs.write(reinterpret_cast<const char*>(digest.data()), 32);
        ofs.flush();
        if (!ofs.good()) {
            err_msg = "Failed while flushing checkpoint payload: " + tmp_path;
            return false;
        }
    }
    if (std::rename(tmp_path.c_str(), path.c_str()) != 0) {
        err_msg = "Atomic rename failed for checkpoint: " + path;
        return false;
    }
    return true;
}

inline bool load_checkpoint_verified(
    const std::string& path,
    const SystemTopology& top,
    std::vector<ReplicaState>& replicas,
    std::vector<uint64_t>& inner_step_counters,
    uint64_t& global_seed,
    uint64_t& outer_step_done,
    uint64_t& exchange_round,
    std::vector<int64_t>& swap_attempts,
    std::vector<int64_t>& swap_accepts,
    MBARAccumulator& mbar_acc,
    std::string& err_msg
) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs) {
        err_msg = "Checkpoint file does not exist or cannot be opened: " + path;
        return false;
    }
    const std::streamsize total_sz = ifs.tellg();
    if (total_sz < 64) {
        err_msg = "Checkpoint file too short (truncated/corrupted): " + path;
        return false;
    }
    ifs.seekg(0, std::ios::beg);

    const size_t payload_sz = static_cast<size_t>(total_sz) - 32;
    std::vector<uint8_t> payload(payload_sz);
    std::array<uint8_t, 32> stored_digest{};

    if (!ifs.read(reinterpret_cast<char*>(payload.data()), static_cast<std::streamsize>(payload_sz)) ||
        !ifs.read(reinterpret_cast<char*>(stored_digest.data()), 32)) {
        err_msg = "I/O error reading checkpoint file: " + path;
        return false;
    }

    Sha256 sha;
    sha.update(payload.data(), payload.size());
    const auto actual_digest = sha.finalize();
    if (actual_digest != stored_digest) {
        err_msg = "Checkpoint SHA-256 verification FAILED (corrupted file): " + path;
        return false;
    }

    size_t pos = 0;
    auto read_bytes = [&](void* dst, size_t sz) -> bool {
        if (pos + sz > payload.size()) return false;
        std::memcpy(dst, payload.data() + pos, sz);
        pos += sz;
        return true;
    };

    char magic[8]{};
    uint32_t version = 0;
    int32_t n_atoms = 0, n_rep = 0;
    uint64_t top_fp = 0;

    if (!read_bytes(magic, 8) || std::memcmp(magic, "ASHMDCHK", 8) != 0) {
        err_msg = "Invalid checkpoint magic header";
        return false;
    }
    if (!read_bytes(&version, sizeof(version)) || version != 1U) {
        err_msg = "Unsupported checkpoint format version";
        return false;
    }
    if (!read_bytes(&n_atoms, sizeof(n_atoms)) || n_atoms != top.n_atoms) {
        err_msg = "Checkpoint atom count does not match current topology";
        return false;
    }
    if (!read_bytes(&n_rep, sizeof(n_rep)) || n_rep <= 0) {
        err_msg = "Invalid replica count in checkpoint";
        return false;
    }
    if (!read_bytes(&top_fp, sizeof(top_fp)) || top_fp != compute_topology_fingerprint(top)) {
        err_msg = "Checkpoint topology fingerprint mismatch (refusing cross-topology resume)";
        return false;
    }
    if (!read_bytes(&global_seed, sizeof(global_seed)) ||
        !read_bytes(&outer_step_done, sizeof(outer_step_done)) ||
        !read_bytes(&exchange_round, sizeof(exchange_round))) {
        err_msg = "Truncated checkpoint metadata";
        return false;
    }

    replicas.resize(n_rep);
    inner_step_counters.resize(n_rep, 0);
    for (int32_t r = 0; r < n_rep; ++r) {
        replicas[r].replica_id = static_cast<uint32_t>(r);
        replicas[r].allocate(n_atoms);
        if (!read_bytes(&inner_step_counters[r], sizeof(uint64_t)) ||
            !read_bytes(&replicas[r].tempering, sizeof(ReplicaTemperingParams)) ||
            !read_bytes(replicas[r].rx.data(), n_atoms * sizeof(double)) ||
            !read_bytes(replicas[r].ry.data(), n_atoms * sizeof(double)) ||
            !read_bytes(replicas[r].rz.data(), n_atoms * sizeof(double)) ||
            !read_bytes(replicas[r].vx.data(), n_atoms * sizeof(double)) ||
            !read_bytes(replicas[r].vy.data(), n_atoms * sizeof(double)) ||
            !read_bytes(replicas[r].vz.data(), n_atoms * sizeof(double))) {
            err_msg = "Truncated replica state in checkpoint";
            return false;
        }
        replicas[r].forces_initialized = false;
    }

    int32_t n_sw = 0;
    if (!read_bytes(&n_sw, sizeof(n_sw)) || n_sw < 0) {
        err_msg = "Corrupted swap counter section";
        return false;
    }
    swap_attempts.assign(n_sw, 0);
    swap_accepts.assign(n_sw, 0);
    if (n_sw > 0) {
        if (!read_bytes(swap_attempts.data(), n_sw * sizeof(int64_t)) ||
            !read_bytes(swap_accepts.data(), n_sw * sizeof(int64_t))) {
            err_msg = "Truncated swap counters";
            return false;
        }
    }

    int32_t mbar_M = 0, mbar_N = 0;
    if (!read_bytes(&mbar_M, sizeof(mbar_M)) || !read_bytes(&mbar_N, sizeof(mbar_N))) {
        err_msg = "Truncated MBAR section";
        return false;
    }
    mbar_acc.reset(std::max(1, mbar_M));
    if (mbar_M > 0) {
        if (!read_bytes(mbar_acc.state_counts.data(), mbar_M * sizeof(int32_t))) {
            err_msg = "Truncated MBAR state counts";
            return false;
        }
    }
    if (mbar_N > 0) {
        mbar_acc.origin_state.resize(mbar_N);
        mbar_acc.u_matrix.resize(static_cast<size_t>(mbar_N) * mbar_M);
        if (!read_bytes(mbar_acc.origin_state.data(), mbar_N * sizeof(int32_t)) ||
            !read_bytes(mbar_acc.u_matrix.data(), static_cast<size_t>(mbar_N) * mbar_M * sizeof(double))) {
            err_msg = "Truncated MBAR potential matrix";
            return false;
        }
    }
    return true;
}

} // namespace ashmd
