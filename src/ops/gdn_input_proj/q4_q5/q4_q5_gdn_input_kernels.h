#pragma once

#include "core/weight.h"
#include "core/tensor.h"
#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_plan.h"

#include "ops/common/int8_proj_launch.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void q4_q5_gdn_input_int8_launch(const Tensor& x, const Weight& qk_weight,
                                 const Weight& value_z_weight, Tensor& qkv, Tensor& z,
                                 const Int8ProjWorkspace& scratch, cudaStream_t stream);

void q4_q5_gdn_input_independent_launch(const Tensor& x, const Weight& qk_weight,
                                        const Weight& value_z_weight, Tensor& qk, Tensor& value,
                                        Tensor& z, cudaStream_t stream);

// The grouped MMA route is the R64/C128 schedule for the multi-token band; the plan
// selects it by schedule id and the kernel no longer takes the id itself.
void q4_q5_gdn_input_grouped_mma_launch(const Tensor& x, const Weight& qk_weight,
                                        const Weight& value_z_weight, Tensor& qkv, Tensor& z,
                                        cudaStream_t stream);

void q4_q5_gdn_input_conv_snapshot_launch(const Tensor& x, const Weight& qk_weight,
                                          const Weight& value_z_weight, const Tensor& conv_weight,
                                          Tensor& conv_states, const Tensor& valid_columns,
                                          const Tensor& initial_slot,
                                          const Tensor& snapshot_base_slot, Tensor& query,
                                          Tensor& key, Tensor& value, Tensor& z,
                                          cudaStream_t stream);

void q4_q5_gdn_input_conv_record_launch(const Tensor& x, const Weight& qk_weight,
                                        const Weight& value_z_weight, const Tensor& conv_weight,
                                        const Tensor& conv_states, const Tensor& valid_columns,
                                        const Tensor& initial_slot, Tensor& conv_record,
                                        Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                                        cudaStream_t stream);

} // namespace ninfer::ops::detail
