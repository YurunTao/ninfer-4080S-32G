#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <utility>

namespace ninfer::pdl {

// Programmatic dependent launch (griddepcontrol, launch_dependents) needs sm_90 or
// newer. This branch targets sm_89, where the same fall back as the sm_86 port
// applies: plain launches in stream order, with no dependency intrinsics emitted.
#if defined(NINFER_SM86) || defined(NINFER_ARCH_SM89)
#define NINFER_PDL_UNAVAILABLE 1
#else
#define NINFER_PDL_UNAVAILABLE 0
#endif

struct LaunchConfig {
    dim3 grid;
    dim3 block;
    std::size_t dynamic_smem_bytes = 0;
    cudaStream_t stream            = nullptr;
};

// Launches a consumer kernel as a programmatic dependent of the immediately preceding producer
// kernel in the same stream. Every consumer control path that reads producer output must first call
// wait_for_dependencies().
template <class... KernelArgs, class... CallArgs>
[[nodiscard]] inline cudaError_t
launch_dependent(const LaunchConfig& launch, void (*kernel)(KernelArgs...), CallArgs&&... args) {
#if NINFER_PDL_UNAVAILABLE
    kernel<<<launch.grid, launch.block, launch.dynamic_smem_bytes, launch.stream>>>(
        std::forward<CallArgs>(args)...);
    return cudaGetLastError();
#else
    cudaLaunchAttribute attribute{};
    attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute.val.programmaticStreamSerializationAllowed = 1;

    cudaLaunchConfig_t config{};
    config.gridDim          = launch.grid;
    config.blockDim         = launch.block;
    config.dynamicSmemBytes = launch.dynamic_smem_bytes;
    config.stream           = launch.stream;
    config.attrs            = &attribute;
    config.numAttrs         = 1;

    return cudaLaunchKernelEx(&config, kernel, std::forward<CallArgs>(args)...);
#endif
}

// Every producer CTA must call this at least once or exit. This enables dependent scheduling but
// does not make producer writes visible to the consumer.
__device__ __forceinline__ void trigger_dependents() {
#if !NINFER_PDL_UNAVAILABLE
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// Call on every consumer control path before its first access to producer-dependent data.
__device__ __forceinline__ void wait_for_dependencies() {
#if !NINFER_PDL_UNAVAILABLE
    cudaGridDependencySynchronize();
#endif
}

} // namespace ninfer::pdl
