// include/basalt/core/card_model_kernels.hpp - whether a card runs the model's own kernels.
//
// Header-only, like gpu_mem_info.hpp. A peer card of another compute capability runs the peer tier's kernels
// that a cross-capability peer build compiles for it, and nothing of the model's own: not a prompt split's
// layers, not the verifier head's rows, not the mixer's state. A plan that would run model kernels there
// declines the card; the card stays what a cross-capability build adds it for, the expert tier.
#pragma once

#include "basalt/core/cuda_cc.hpp"

#include <cuda_runtime_api.h>

namespace basalt::core {

/// Whether device `dev`'s compute capability is the one this build compiled the model's kernels for;
/// false when the runtime cannot say, so an unknown card is not planned onto.
inline bool card_runs_model_kernels(int dev) {
    int major = 0, minor = 0;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess) {
        (void) cudaGetLastError();
        return false;
    }
    return cuda_cc_is_ours(major, minor);
}

}  // namespace basalt::core
