// include/basalt/kernels/l2_pf.cuh - pull a global byte range into L2 without touching registers.
//
// sm_90+ asks the copy engine once (cp.async.bulk.prefetch.L2.global); Ada has no bulk prefetch, so the same
// range is walked one 128-byte line at a time with prefetch.global.L2.  Both are
// fire-and-forget requests for L2 only: a caller that needs the bytes landed reads a word of the range after
// issuing, exactly as the read-ahead loops that call this do.  A caller issues for itself (one thread) or the
// threads of a block must walk disjoint ranges: nothing here deduplicates.  `bytes` must be a multiple of 16
// (the bulk form's requirement; every caller keeps it), and the range must be inside a live mapping.
#pragma once

#include <cstdint>

namespace basalt::kernels {

__device__ __forceinline__ void pf_l2(const void* src, uint64_t bytes) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(src), "r"((unsigned) bytes) : "memory");
#elif defined(__CUDA_ARCH__)
    const char* p = (const char*) src;
    for (const char* q = p; q < p + bytes; q += 128)
        asm volatile("prefetch.global.L2 [%0];" ::"l"(q) : "memory");
#endif
}

}  // namespace basalt::kernels
