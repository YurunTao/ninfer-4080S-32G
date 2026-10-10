#include "core/weight.h"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"

#include "core/device.h"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_output.cuh"
#include "ops/linear/nvfp4/nvfp4_schedule.cuh"
#include "ops/linear/nvfp4/nvfp4_template_launch.cuh"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_a4_tma_launch.h"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

// Build-level capability switch; see cmake/NinferTargets.cmake.
#if !defined(NINFER_ARCH_SM89)
#define NINFER_ARCH_SM89 0
#endif

namespace ninfer::ops::detail {
namespace {

using Geometry = Nvfp4N14336K5120;

using M32N64            = Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>;
using M32N128           = Nvfp4A4MmaSchedule<32, 128, 256, 2, 4, 2, 1>;
using M64N128           = Nvfp4A4MmaSchedule<64, 128, 256, 4, 2, 2, 1>;
using M128N128Pipelined = Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 2, 1>;
using M128N128Resident  = Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 1, 2>;

// This projection selects its own route, so the layout the quantizer writes below must be derived
// from the same predicate; the two are read together at the call site for that reason.
constexpr bool uses_tma(std::int32_t tokens) { return tokens >= 512; }

template <class Schedule>
void launch_gemm(const Weight& weight, Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                 Nvfp4A4Workspace workspace, std::int32_t tokens, cudaStream_t stream) {
    launch_nvfp4_a4_mma<Nvfp4ScheduleInstance<Schedule, Geometry::kInputRows>>(
        nvfp4_a4_operands(weight, workspace, tokens, Nvfp4ScaleLayout::RowMajor),
        Nvfp4AttentionInputOutput{
            static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
            static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(v.data)},
        LinearIdentityEpilogue{}, stream);
}

} // namespace

void nvfp4_attn_input_a4_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                Tensor& k, Tensor& v, Nvfp4A4Workspace workspace,
                                cudaStream_t stream) {
#if NINFER_ARCH_SM89
    // The A4 route is built from Blackwell block-scale MMA and Hopper-or-newer TMA.
    // Neither exists below sm_90, so the route is not compiled on this branch and
    // the caller gets an explicit rejection instead of a silent numerical fallback.
    (void)x;
    (void)weight;
    (void)q;
    (void)gate;
    (void)k;
    (void)v;
    (void)workspace;
    (void)stream;
    throw std::invalid_argument(
        "NVFP4 A4 attention input requires an sm_120a GPU; this build targets sm_89");
#else
    const std::int32_t tokens = x.ne[1];
    const auto layout =
        uses_tma(tokens) ? (tokens < 1024 ? Nvfp4ScaleLayout::Tiled128 : Nvfp4ScaleLayout::Tiled256)
                         : Nvfp4ScaleLayout::RowMajor;
    launch_nvfp4_a4_quantize(x, weight, workspace, layout, stream);
    if (uses_tma(tokens)) {
        launch_nvfp4_a4_tma_attention(
            nvfp4_a4_operands(weight, workspace, tokens, layout),
            static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(gate.data),
            static_cast<__nv_bfloat16*>(k.data), static_cast<__nv_bfloat16*>(v.data), stream);
    } else if (tokens <= 64) {
        launch_gemm<M32N64>(weight, q, gate, k, v, workspace, tokens, stream);
    } else if (tokens <= 96) {
        launch_gemm<M32N128>(weight, q, gate, k, v, workspace, tokens, stream);
    } else if (tokens <= 128) {
        launch_gemm<M128N128Pipelined>(weight, q, gate, k, v, workspace, tokens, stream);
    } else if (tokens <= 192) {
        launch_gemm<M64N128>(weight, q, gate, k, v, workspace, tokens, stream);
    } else if (tokens <= 384) {
        launch_gemm<M128N128Resident>(weight, q, gate, k, v, workspace, tokens, stream);
    } else if (tokens <= 512) {
        launch_gemm<M128N128Pipelined>(weight, q, gate, k, v, workspace, tokens, stream);
    } else {
        launch_gemm<M128N128Resident>(weight, q, gate, k, v, workspace, tokens, stream);
    }
#endif // NINFER_ARCH_SM89
}

} // namespace ninfer::ops::detail
