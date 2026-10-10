// sm_89 build shim for the NVFP4 A4 execution family.
//
// The NVFP4 A4 kernels are compiled from src/ops/**/nvfp4/* into the
// ninfer_nvfp4_non_rdc archive on the sm_120a target only. They rely on
// Blackwell block-scale MMA (kind::mxf4nvf4, .block_scale, .scale_vec::4X) and
// on Hopper-or-newer TMA/mbarrier primitives, none of which ptxas accepts for
// sm_89. src/ops/CMakeLists.txt therefore degrades ninfer_nvfp4_non_rdc to an
// interface library when NINFER_ARCH_SM89 is set, and this translation unit
// provides the six launcher entry points those kernels would have defined.
//
// Every entry point rejects: NVFP4 A4 execution is a Blackwell-only capability
// of this engine. Rejecting here keeps artifact loading explicit and gives the
// caller a diagnosable failure instead of a silent numerical fallback.

#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_a4_tma_launch.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_a4_tma_launch.h"
#include "ops/linear/nvfp4/nvfp4_a4_tma_launch.h"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_a4_tma_launch.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_a4_tma_launch.h"
#include "ops/softmax_attention/dense/causal_cache/nvfp4/tiled_launch.h"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

[[noreturn]] void reject_nvfp4_a4() {
    throw std::runtime_error(
        "NVFP4 A4 execution requires an sm_120a GPU; this build targets sm_89");
}

} // namespace

void launch_nvfp4_a4_tma_linear(Nvfp4GeometryId, const Nvfp4A4Operands&, __nv_bfloat16*,
                                cudaStream_t) {
    reject_nvfp4_a4();
}

void launch_nvfp4_a4_tma_attention(const Nvfp4A4Operands&, __nv_bfloat16*, __nv_bfloat16*,
                                   __nv_bfloat16*, __nv_bfloat16*, cudaStream_t) {
    reject_nvfp4_a4();
}

void launch_nvfp4_a4_tma_gdn(const Nvfp4A4Operands&, __nv_bfloat16*, __nv_bfloat16*,
                             cudaStream_t) {
    reject_nvfp4_a4();
}

void launch_nvfp4_a4_tma_linear_add(const Nvfp4A4Operands&, __nv_bfloat16*, cudaStream_t) {
    reject_nvfp4_a4();
}

void launch_nvfp4_linear_swiglu_a4_tma(const Nvfp4A4Operands&, __nv_bfloat16*, cudaStream_t) {
    reject_nvfp4_a4();
}

void nvfp4_kv_tiled_attention(const CausalAttentionOperands&, Nvfp4KvReadView, cudaStream_t) {
    reject_nvfp4_a4();
}

} // namespace ninfer::ops::detail
