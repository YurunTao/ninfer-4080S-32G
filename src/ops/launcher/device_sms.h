#pragma once

// Launcher-level device fact: the resident SM count of the current device.
//
// Upstream launchers hard-coded the 170 SMs of the RTX 5090 for wave/prefetch crossings; that
// literal is wrong on every other part and the comments said so ("not portable"). This helper
// queries the real count once per process. Callers run on the engine's bound device, so the
// first query happens on the device whose count matters.

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

[[nodiscard]] inline int device_multiprocessor_count() noexcept {
    static const int count = [] {
        int device  = 0;
        int value   = 0;
        if (cudaGetDevice(&device) != cudaSuccess ||
            cudaDeviceGetAttribute(&value, cudaDevAttrMultiProcessorCount, device) != cudaSuccess ||
            value <= 0) {
            return 128;  // conservative mid-range fallback
        }
        return value;
    }();
    return count;
}

}  // namespace ninfer::ops::detail
