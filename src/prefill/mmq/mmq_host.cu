// The host side of the MMQ kernels: error reporting, the properties of each device and the cache of scratch buffers
// that the stream-k fixup of a launch takes from, and the explicit load of a kernel.
#include "mmq_common.cuh"
#include "basalt/core/device.hpp"
#include "basalt/core/on_device.hpp"

#include <cuda.h>

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>
#include "basalt/log.hpp"

[[noreturn]] void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    basalt::log::error("basalt", "mmq: %s: %s\n  in %s at %s:%d", msg, stmt, func, file, line);
    std::abort();
}

[[noreturn]] void ggml_cuda_oom(const cudaError_t err, const long long asked_bytes, const char * stmt,
                                const char * func, const char * file, const int line) {
    int dev = -1;
    size_t free_b = 0, total_b = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) { dev = -1; }
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) { free_b = 0; }
    (void) cudaGetLastError();
    char msg[192];
    snprintf(msg, sizeof msg, "%s (asked %lld B, device %d, %llu MiB free)", cudaGetErrorString(err),
             asked_bytes, dev, (unsigned long long) (free_b >> 20));
    ggml_cuda_error(stmt, func, file, line, msg);
}

int ggml_cuda_get_device() {
    int id = 0;
    CUDA_CHECK(cudaGetDevice(&id));
    return id;
}

void ggml_cuda_func_load(const void * symbol) {
    using Load = CUresult (CUDAAPI *)(CUfunction);
    static const Load load = [] {
        void * p = nullptr;
        cudaDriverEntryPointQueryResult q{};
        if (cudaGetDriverEntryPointByVersion("cuFuncLoad", &p, 12040, cudaEnableDefault, &q) != cudaSuccess ||
            q != cudaDriverEntryPointSuccess) {
            (void) cudaGetLastError();
            p = nullptr;
        }
        return (Load) p;
    }();
    cudaFunction_t f = nullptr;
    CUDA_CHECK(cudaGetFuncBySymbol(&f, symbol));
    if (load == nullptr || load((CUfunction) f) != CUDA_SUCCESS)
        GGML_ABORT("cannot load a kernel (cuFuncLoad)");
}

const ggml_cuda_device_info & ggml_cuda_info() {
    static ggml_cuda_device_info info = [] {
        ggml_cuda_device_info in = {};
        int n = 0;
        if (cudaGetDeviceCount(&n) != cudaSuccess) n = 0;
        n = n > GGML_CUDA_MAX_DEVICES ? GGML_CUDA_MAX_DEVICES : n;
        in.device_count = n;
        for (int id = 0; id < n; ++id) {
            // cudaDeviceGetAttribute is ABI-stable across runtime versions, a cudaDeviceProp is not.
            auto attr = [id](cudaDeviceAttr a) {
                int v = 0;
                CUDA_CHECK(cudaDeviceGetAttribute(&v, a, id));
                return v;
            };
            auto & d = in.devices[id];
            d.smpbo = (size_t) attr(cudaDevAttrMaxSharedMemoryPerBlockOptin);
            d.nsm = basalt::core::plan_sms(id);   // the stream-k split: the planning card's, so a stage computes its bits
            d.warp_size = attr(cudaDevAttrWarpSize);
        }
        return in;
    }();
    return info;
}

namespace {
// Buffers are kept and reused: MMQ asks for the same few sizes every launch (its stream-k fixup tiles).
struct CachingPool : ggml_cuda_pool {
    int device;
    explicit CachingPool(int dev) : device(dev) {}
    struct Buf { void * p; size_t size; bool used; };
    std::vector<Buf> bufs;
    std::mutex mu;
    void * alloc(size_t size, size_t * actual_size) override {
        std::lock_guard<std::mutex> lk(mu);
        for (auto & b : bufs)
            if (!b.used && b.size >= size) { b.used = true; *actual_size = b.size; return b.p; }
        const basalt::core::OnDevice od(device);
        void * p = nullptr;
        const cudaError_t e = cudaMalloc(&p, size);
        if (e != cudaSuccess) { ggml_cuda_oom(e, (long long) size, "cudaMalloc(&p, size)", __func__, __FILE__, __LINE__); }
        bufs.push_back({p, size, true});
        *actual_size = size;
        return p;
    }
    void free(void * ptr, size_t) override {
        std::lock_guard<std::mutex> lk(mu);
        for (auto & b : bufs)
            if (b.p == ptr) { b.used = false; return; }
    }
    ~CachingPool() override {
        const basalt::core::OnDevice od(device);
        for (auto & b : bufs) cudaFree(b.p);
    }
};
}  // namespace

std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(int device) {
    return std::unique_ptr<ggml_cuda_pool>(new CachingPool(device));
}

ggml_backend_cuda_context::~ggml_backend_cuda_context() {}
