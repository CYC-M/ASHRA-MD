#pragma once

// Portable SIMD abstraction layer for ASH-MD:
// - Targets AMD EPYC AVX2 + FMA (256-bit, 4x float64) on Instabox compute nodes
// - Targets Apple Silicon ARM64 NEON (128-bit, 2x float64) on macOS M-series
// - Clean scalar fallback for any other architecture

#include <cmath>
#include <cstdint>

#if defined(__AVX2__)
    #include <immintrin.h>
    #define ASHMD_SIMD_BACKEND "AVX2_256"
    #define ASHMD_SIMD_WIDTH_F64 4
#elif defined(__ARM_NEON) || defined(__aarch64__)
    #include <arm_neon.h>
    #define ASHMD_SIMD_BACKEND "ARM64_NEON_128"
    #define ASHMD_SIMD_WIDTH_F64 2
#else
    #define ASHMD_SIMD_BACKEND "SCALAR_64"
    #define ASHMD_SIMD_WIDTH_F64 1
#endif

namespace ashmd::simd {

struct Vec4d {
#if defined(__AVX2__)
    __m256d v;
    inline Vec4d() = default;
    inline explicit Vec4d(double s) : v(_mm256_set1_pd(s)) {}
    inline explicit Vec4d(__m256d x) : v(x) {}
    inline Vec4d(double a, double b, double c, double d) : v(_mm256_setr_pd(a, b, c, d)) {}
    static inline Vec4d loadu(const double* p) { return Vec4d(_mm256_loadu_pd(p)); }
    inline void storeu(double* p) const { _mm256_storeu_pd(p, v); }
    inline Vec4d operator+(const Vec4d& o) const { return Vec4d(_mm256_add_pd(v, o.v)); }
    inline Vec4d operator-(const Vec4d& o) const { return Vec4d(_mm256_sub_pd(v, o.v)); }
    inline Vec4d operator*(const Vec4d& o) const { return Vec4d(_mm256_mul_pd(v, o.v)); }
    inline Vec4d operator/(const Vec4d& o) const { return Vec4d(_mm256_div_pd(v, o.v)); }
    static inline Vec4d fmadd(const Vec4d& a, const Vec4d& b, const Vec4d& c) {
    #if defined(__FMA__)
        return Vec4d(_mm256_fmadd_pd(a.v, b.v, c.v));
    #else
        return Vec4d(_mm256_add_pd(_mm256_mul_pd(a.v, b.v), c.v));
    #endif
    }
    static inline Vec4d sqrt(const Vec4d& a) { return Vec4d(_mm256_sqrt_pd(a.v)); }
    inline double hsum() const {
        alignas(32) double tmp[4];
        _mm256_store_pd(tmp, v);
        return (tmp[0] + tmp[1]) + (tmp[2] + tmp[3]);
    }
#elif defined(__ARM_NEON) || defined(__aarch64__)
    float64x2_t lo, hi;
    inline Vec4d() = default;
    inline explicit Vec4d(double s) : lo(vdupq_n_f64(s)), hi(vdupq_n_f64(s)) {}
    inline Vec4d(float64x2_t l, float64x2_t h) : lo(l), hi(h) {}
    inline Vec4d(double a, double b, double c, double d) {
        double arr[4] = {a, b, c, d};
        lo = vld1q_f64(arr);
        hi = vld1q_f64(arr + 2);
    }
    static inline Vec4d loadu(const double* p) {
        return Vec4d(vld1q_f64(p), vld1q_f64(p + 2));
    }
    inline void storeu(double* p) const {
        vst1q_f64(p, lo);
        vst1q_f64(p + 2, hi);
    }
    inline Vec4d operator+(const Vec4d& o) const { return Vec4d(vaddq_f64(lo, o.lo), vaddq_f64(hi, o.hi)); }
    inline Vec4d operator-(const Vec4d& o) const { return Vec4d(vsubq_f64(lo, o.lo), vsubq_f64(hi, o.hi)); }
    inline Vec4d operator*(const Vec4d& o) const { return Vec4d(vmulq_f64(lo, o.lo), vmulq_f64(hi, o.hi)); }
    inline Vec4d operator/(const Vec4d& o) const { return Vec4d(vdivq_f64(lo, o.lo), vdivq_f64(hi, o.hi)); }
    static inline Vec4d fmadd(const Vec4d& a, const Vec4d& b, const Vec4d& c) {
        return Vec4d(vfmaq_f64(c.lo, a.lo, b.lo), vfmaq_f64(c.hi, a.hi, b.hi));
    }
    static inline Vec4d sqrt(const Vec4d& a) { return Vec4d(vsqrtq_f64(a.lo), vsqrtq_f64(a.hi)); }
    inline double hsum() const {
        float64x2_t s = vaddq_f64(lo, hi);
        return vaddvq_f64(s);
    }
#else
    double d[4];
    inline Vec4d() = default;
    inline explicit Vec4d(double s) : d{s, s, s, s} {}
    inline Vec4d(double a, double b, double c, double e) : d{a, b, c, e} {}
    static inline Vec4d loadu(const double* p) { return Vec4d(p[0], p[1], p[2], p[3]); }
    inline void storeu(double* p) const { p[0]=d[0]; p[1]=d[1]; p[2]=d[2]; p[3]=d[3]; }
    inline Vec4d operator+(const Vec4d& o) const { return Vec4d(d[0]+o.d[0], d[1]+o.d[1], d[2]+o.d[2], d[3]+o.d[3]); }
    inline Vec4d operator-(const Vec4d& o) const { return Vec4d(d[0]-o.d[0], d[1]-o.d[1], d[2]-o.d[2], d[3]-o.d[3]); }
    inline Vec4d operator*(const Vec4d& o) const { return Vec4d(d[0]*o.d[0], d[1]*o.d[1], d[2]*o.d[2], d[3]*o.d[3]); }
    inline Vec4d operator/(const Vec4d& o) const { return Vec4d(d[0]/o.d[0], d[1]/o.d[1], d[2]/o.d[2], d[3]/o.d[3]); }
    static inline Vec4d fmadd(const Vec4d& a, const Vec4d& b, const Vec4d& c) {
        return Vec4d(a.d[0]*b.d[0]+c.d[0], a.d[1]*b.d[1]+c.d[1], a.d[2]*b.d[2]+c.d[2], a.d[3]*b.d[3]+c.d[3]);
    }
    static inline Vec4d sqrt(const Vec4d& a) {
        return Vec4d(std::sqrt(a.d[0]), std::sqrt(a.d[1]), std::sqrt(a.d[2]), std::sqrt(a.d[3]));
    }
    inline double hsum() const { return (d[0] + d[1]) + (d[2] + d[3]); }
#endif
};

inline const char* backend_name() noexcept {
    return ASHMD_SIMD_BACKEND;
}

} // namespace ashmd::simd
