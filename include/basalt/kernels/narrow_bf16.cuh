// include/basalt/kernels/narrow_bf16.cuh - FP8 E4M3 and FP4 E2M1 codes as bf16x2 for a kernel's mma operands.  Device
// code only (no kernel): include from a .cu.
//
// Every E4M3 and E2M1 value is exact in bf16.  `cvt.rn.bf16x2.e4m3x2` and `cvt.rn.bf16x2.e2m1x2` (one instruction
// each) need PTX ISA 9.2, CUDA 13.2: the ptxas of CUDA 13.0 and 13.1 rejects them.  There the codes go to f16x2
// (`cvt.rn.f16x2.*`, also exact) and each half widens through f32 to bf16, which gives the same bits, NaNs included.
// sm_89 has neither cvt (nor `mul.rn.bf16x2`), so there an E2M1 code's bits are built from its fields with integer
// arithmetic and the scaling runs in f32, again the same bits.  tests/kernels/narrow_bf16_parity.cu compares the ways
// on every input, on each of the two cards.
#pragma once

#include <cuda_runtime.h>

#if defined(__CUDACC_VER_MAJOR__) && \
    (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
#define BASALT_CVT_BF16X2_NARROW 1
#else
#define BASALT_CVT_BF16X2_NARROW 0
#endif

namespace basalt::kernels {

/// f16x2 as bf16x2, for values bf16 holds exactly (each half through f32)
__device__ __forceinline__ unsigned f16x2_bf16x2(unsigned f) {
    unsigned h;
    asm("{\n"
        ".reg .f16 l, u;\n"
        ".reg .f32 a, b;\n"
        "mov.b32 {l, u}, %1;\n"
        "cvt.f32.f16 a, l;\n"
        "cvt.f32.f16 b, u;\n"
        "cvt.rn.bf16x2.f32 %0, b, a;\n"
        "}\n"
        : "=r"(h)
        : "r"(f));
    return h;
}

/// two E4M3 bytes (the low 16 bits of x, the low byte the lower half) as bf16x2 through f16x2
__device__ __forceinline__ unsigned e4m3x2_bf16x2_via_f16(unsigned x) {
    unsigned f;
    const unsigned short v = (unsigned short) x;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(f) : "h"(v));
    return f16x2_bf16x2(f);
}

/// the four bytes of w, each two E2M1 codes (the low nibble the lower half), as bf16x2 through f16x2
__device__ __forceinline__ void e2m1x8_bf16x2_via_f16(unsigned w, unsigned (&h)[4]) {
    unsigned f[4];
    asm("{\n"
        ".reg .b8 b0, b1, b2, b3;\n"
        "mov.b32 {b0, b1, b2, b3}, %4;\n"
        "cvt.rn.f16x2.e2m1x2 %0, b0;\n"
        "cvt.rn.f16x2.e2m1x2 %1, b1;\n"
        "cvt.rn.f16x2.e2m1x2 %2, b2;\n"
        "cvt.rn.f16x2.e2m1x2 %3, b3;\n"
        "}\n"
        : "=r"(f[0]), "=r"(f[1]), "=r"(f[2]), "=r"(f[3])
        : "r"(w));
#pragma unroll
    for (int i = 0; i < 4; ++i) h[i] = f16x2_bf16x2(f[i]);
}

/// the bf16 bits of one E2M1 code, built from its fields: the magnitudes are f32-normal all of them, so the exponent is
/// f32's (its bias) and the mantissa one bit, the sign in the bf16 sign bit
__device__ __forceinline__ unsigned e2m1_bf16_bits(unsigned c) {
    const unsigned e = (c >> 1) & 3u, m = c & 1u;
    return ((c >> 3) << 15) | (e ? ((e + 126u) << 7) | (m << 6) : m * 0x3f00u);
}

/// the f16 bits of one E2M1 code, the same fields on f16's exponent
__device__ __forceinline__ unsigned e2m1_f16_bits(unsigned c) {
    const unsigned e = (c >> 1) & 3u, m = c & 1u;
    return ((c >> 3) << 15) | (e ? ((e + 14u) << 10) | (m << 9) : m * 0x3800u);
}

/// the four bytes of w, each two E2M1 codes (the low nibble the lower half), as bf16x2 times the bf16x2 scale sc: the
/// codes' bits from the fields, each half's product in f32 (a NaN scale gives a NaN half there as in bf16x2), repacked
/// with cvt.rn.bf16x2.f32 - the sm_89 body, exact like the direct one for the same reason
__device__ __forceinline__ void e2m1x8_bf16x2_scaled_via_f32(unsigned w, unsigned sc, unsigned (&h)[4]) {
    const float sa = __uint_as_float((sc & 0xffffu) << 16), sb = __uint_as_float(sc & 0xffff0000u);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const unsigned b = (w >> (8 * i)) & 0xffu;
        const float lo = __fmul_rn(__uint_as_float(e2m1_bf16_bits(b & 0xfu) << 16), sa);
        const float hi = __fmul_rn(__uint_as_float(e2m1_bf16_bits(b >> 4) << 16), sb);
        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(h[i]) : "f"(hi), "f"(lo));
    }
}

/// the same through f16x2 (the codes' f16 bits from the fields, mul.rn.f16x2, each half widened back through f32), all
/// of it on sm_89 - the parity test's other route where the Blackwell one (the cvt above times mul.rn.bf16x2) is absent
__device__ __forceinline__ void e2m1x8_bf16x2_scaled_via_f16(unsigned w, unsigned sc, unsigned (&h)[4]) {
    unsigned short l, u;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(l) : "f"(__uint_as_float((sc & 0xffffu) << 16)));
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(u) : "f"(__uint_as_float(sc & 0xffff0000u)));
    const unsigned s = (unsigned) l | ((unsigned) u << 16);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const unsigned b = (w >> (8 * i)) & 0xffu;
        const unsigned t = e2m1_f16_bits(b & 0xfu) | (e2m1_f16_bits(b >> 4) << 16);
        unsigned m;
        asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(m) : "r"(t), "r"(s));
        h[i] = f16x2_bf16x2(m);
    }
}

/// two E4M3 bytes (the low 16 bits of x, the low byte the lower half) as bf16x2 (exact)
__device__ __forceinline__ unsigned e4m3x2_bf16x2(unsigned x) {
#if BASALT_CVT_BF16X2_NARROW && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
    unsigned h;
    const unsigned short v = (unsigned short) x;
    asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(h) : "h"(v));
    return h;
#else
    return e4m3x2_bf16x2_via_f16(x);
#endif
}

/// the four bytes of w, each two E2M1 codes (the low nibble the lower half), as bf16x2 times the bf16x2 scale sc (exact:
/// a code times a UE4M3 scale has at most five significant bits)
__device__ __forceinline__ void e2m1x8_bf16x2_scaled(unsigned w, unsigned sc, unsigned (&h)[4]) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
#if BASALT_CVT_BF16X2_NARROW
    asm("{\n"
        ".reg .b8 b0, b1, b2, b3;\n"
        ".reg .b32 t0, t1, t2, t3;\n"
        "mov.b32 {b0, b1, b2, b3}, %4;\n"
        "cvt.rn.bf16x2.e2m1x2 t0, b0;\n"
        "cvt.rn.bf16x2.e2m1x2 t1, b1;\n"
        "cvt.rn.bf16x2.e2m1x2 t2, b2;\n"
        "cvt.rn.bf16x2.e2m1x2 t3, b3;\n"
        "mul.rn.bf16x2 %0, t0, %5;\n"
        "mul.rn.bf16x2 %1, t1, %5;\n"
        "mul.rn.bf16x2 %2, t2, %5;\n"
        "mul.rn.bf16x2 %3, t3, %5;\n"
        "}\n"
        : "=r"(h[0]), "=r"(h[1]), "=r"(h[2]), "=r"(h[3])
        : "r"(w), "r"(sc));
#else
    unsigned t[4];
    e2m1x8_bf16x2_via_f16(w, t);
#pragma unroll
    for (int i = 0; i < 4; ++i) asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(h[i]) : "r"(t[i]), "r"(sc));
#endif
#else
    e2m1x8_bf16x2_scaled_via_f32(w, sc, h);
#endif
}

}  // namespace basalt::kernels
