#pragma once

#include "ops/common/mbarrier.cuh"
#include "ops/common/memory.cuh"
#include "ops/linear/bf16/bf16_mma_common.cuh"
#include "ops/linear/bf16/bf16_operands.h"
#include "ops/common/token_slices.h"
#include "ops/common/math.h"
#include <cuda.h>
#include <stdexcept>
#include <string>

#if !defined(NINFER_ARCH_SM89)
#define NINFER_ARCH_SM89 0
#endif

namespace ninfer::ops::detail {

#if !NINFER_ARCH_SM89

struct alignas(128) Bf16TmaDescriptors {
    CUtensorMap weight;
    CUtensorMap activation;
};

template <class Schedule>
inline CUtensorMap bf16_tma_map(const __nv_bfloat16* pointer, int rows, int k, int block_rows,
                                int block_k) {
    // Factor K into 64-element sectors. The contiguous 128-byte dimension matches the
    // hardware swizzle while the next box dimension permits larger K tiles without repacking.
    CUtensorMap result{};
    std::uint64_t dimensions[]{64, static_cast<std::uint64_t>(k / 64),
                               static_cast<std::uint64_t>(rows)};
    std::uint64_t strides[]{128, static_cast<std::uint64_t>(k) * 2};
    std::uint32_t box[]{64, static_cast<std::uint32_t>(block_k / 64),
                        static_cast<std::uint32_t>(block_rows)};
    if constexpr (bf16_predicated_k<Schedule>) {
        // A logical 2D K axis lets TMA zero-fill its partial final sector, preserving row stride.
        static_assert(Schedule::kBlockK == 64, "K-tail TMA requires a 128-byte inner box");
        dimensions[0] = static_cast<std::uint64_t>(k);
        dimensions[1] = static_cast<std::uint64_t>(rows);
        strides[0]    = static_cast<std::uint64_t>(k) * 2;
        box[1]        = static_cast<std::uint32_t>(block_rows);
    }
    const std::uint32_t steps[]{1, 1, 1};
    const auto status = cuTensorMapEncodeTiled(
        &result, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, bf16_predicated_k<Schedule> ? 2 : 3,
        const_cast<__nv_bfloat16*>(pointer), dimensions, strides, box, steps,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (status != CUDA_SUCCESS) {
        const char* name = nullptr;
        (void)cuGetErrorName(status, &name);
        throw std::runtime_error(std::string("BF16 TMA descriptor: ") +
                                 (name ? name : "CUDA error"));
    }
    return result;
}

template <class Schedule>
Bf16TmaDescriptors make_bf16_tma_descriptors(const Bf16A16Operands& p) {
    validate_bf16_operands<Schedule>(p);
    if ((!bf16_predicated_rows<Schedule> && p.rows % Schedule::kBlockRows) ||
        p.k % (bf16_predicated_k<Schedule> ? 8 : Schedule::kBlockK))
        throw std::invalid_argument("BF16 TMA requires compatible row/K tiles");
    return {bf16_tma_map<Schedule>(p.weight, p.rows, p.k, Schedule::kBlockRows, Schedule::kBlockK),
            bf16_tma_map<Schedule>(p.x, p.tokens, p.k, Schedule::kBlockTokens, Schedule::kBlockK)};
}

template <class Schedule>
__device__ __forceinline__ void bf16_tma_load(void* destination, const CUtensorMap* map,
                                              int k_sector, int row, std::uint64_t* barrier) {
    if constexpr (bf16_predicated_k<Schedule>) {
        asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
                     "[%0], [%1, {%2, %3}], [%4];"
                     :
                     : "r"(smem_addr(destination)), "l"(map), "r"(k_sector * 64), "r"(row),
                       "r"(smem_addr(barrier))
                     : "memory");
    } else {
        asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
                     "[%0], [%1, {%2, %3, %4}], [%5];"
                     :
                     : "r"(smem_addr(destination)), "l"(map), "r"(0), "r"(k_sector), "r"(row),
                       "r"(smem_addr(barrier))
                     : "memory");
    }
}

#endif // !NINFER_ARCH_SM89

template <class Schedule, class Epilogue>
inline constexpr int bf16_tma_scratch_bytes =
    ((Schedule::kTensorBytes > bf16_epilogue_bytes<Schedule, Epilogue>
          ? Schedule::kTensorBytes
          : bf16_epilogue_bytes<Schedule, Epilogue>)+127) /
    128 * 128;

#if NINFER_ARCH_SM89

// sm_89 route: the same tile, pipeline depth and MMA schedule, but the tiles are
// staged by cp.async from plain row-major global rows instead of TMA descriptors
// and mbarrier transactions. Every thread participates in both staging and MMA,
// so no producer warp group and no sm_90 mbarrier are required. cp.async copies
// complete in commit order, which lets one wait_group cover the whole pipeline.

// Copies one (rows x block_k) tile of bf16 elements from a row-major global matrix
// whose row pitch is `pitch` elements. `block_rows_active` clamps predicated row
// tails; the k axis is predicated per row so no element outside the matrix is read.
template <class Schedule>
__device__ __forceinline__ void bf16_cpasync_tile(const __nv_bfloat16* source, int pitch,
                                                  int row_begin, int k_begin, int rows_available,
                                                  __nv_bfloat16* destination) {
    constexpr int BR = Schedule::kBlockRows, BT = Schedule::kBlockTokens, BK = Schedule::kBlockK;
    constexpr int kVec = 8; // 16-byte cp.async granularity for bf16
    static_assert(BK % kVec == 0);
    const int k_tiles    = BK / kVec;
    const bool predicated = bf16_predicated_k<Schedule> || bf16_predicated_rows<Schedule>;
    const int rows        = predicated ? rows_available : BR;
    for (int index = threadIdx.x; index < rows * k_tiles; index += blockDim.x) {
        const int row = index / k_tiles, col = (index % k_tiles) * kVec;
        const int k   = k_begin + col;
        if (bf16_predicated_k<Schedule> && k >= pitch) {
            // Zero-fill keeps the MMA reading defined values past the K tail.
            cp_async_zfill<16>(destination + row * BK + col, source, 0);
        } else {
            cp_async<16>(destination + row * BK + col,
                         source + static_cast<std::int64_t>(row_begin + row) * pitch + k);
        }
    }
}

template <class Schedule, bool FullTokens, class Output, class Epilogue>
__global__
__launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void bf16_a16_tma_mma_kernel(
    Output output, Epilogue epilogue, int rows, int input_rows, int token_offset, int count,
    const __nv_bfloat16* weight, const __nv_bfloat16* activation) {
    constexpr int BR = Schedule::kBlockRows, BT = Schedule::kBlockTokens, BK = Schedule::kBlockK;
    constexpr int S   = Schedule::kStages;
    const int K       = Schedule::kStaticK ? Schedule::kStaticK : input_rows;
    const int tiles_r = bf16_predicated_rows<Schedule> ? (rows + BR - 1) / BR : rows / BR;
    const int tiles_t = (count + BT - 1) / BT;
    int tile_r, tile_t;
    bf16_mma_tile_coordinates<Schedule>(blockIdx.x, tiles_r, tiles_t, tile_r, tile_t);
    const int row_begin = tile_r * BR, token_begin = token_offset + tile_t * BT;
    const int tiles_k   = K / BK + (bf16_predicated_k<Schedule> && K % BK != 0);
    const int row_count = (rows - row_begin < BR) ? rows - row_begin : BR;

    extern __shared__ __align__(128) unsigned char storage[];
    auto* a = reinterpret_cast<__nv_bfloat16*>(storage);
    auto* b = a + S * BR * BK;

    const auto stage_tile = [&](int kt, int stage) {
        const int k_begin = kt * BK;
        const int t_count = (token_begin + BT > token_offset + count) ? token_offset + count - token_begin
                                                                     : BT;
        bf16_cpasync_tile<Schedule>(weight, K, row_begin, k_begin, row_count, a + stage * BR * BK);
        bf16_cpasync_tile<Schedule>(activation, K, token_begin, k_begin, t_count,
                                    b + stage * BT * BK);
        cp_commit();
    };

#pragma unroll
    for (int kt = 0; kt < (tiles_k < S ? tiles_k : S); ++kt) { stage_tile(kt, kt % S); }

    const int tid = threadIdx.x, warp = tid / 32, lane = tid & 31;
    float accum[Schedule::kMmaRows][Schedule::kMmaTokens][4] = {};
#pragma unroll Schedule::kConsumerKUnroll
    for (int kt = 0; kt < tiles_k; ++kt) {
        const int stage = kt % S;
        // The group issued S tiles ago has completed once S-1 newer groups are in flight.
        if (kt + S < tiles_k) {
            cp_wait<S - 1>();
            stage_tile(kt + S, stage);
        } else {
            cp_wait<0>();
        }
        __syncthreads();
        bf16_mma_compute_stage<Schedule>(a + stage * BR * BK, b + stage * BT * BK, accum, warp, lane);
        __syncthreads();
    }
    bf16_finish_mma_tile<Schedule, FullTokens>(output, epilogue, storage, accum, row_begin,
                                               token_begin, rows, token_offset + count, warp, lane);
}

template <class Schedule, class Output, class Epilogue>
void launch_bf16_a16_tma_mma(const Bf16A16Operands& p, Output output, Epilogue epilogue,
                             cudaStream_t stream) {
    validate_bf16_operands<Schedule>(p);
    for_each_token_slice(p.tokens, Schedule::kBlockTokens, [&](int offset, int count) {
        const auto blocks = static_cast<std::int64_t>(bf16_predicated_rows<Schedule>
                                                          ? div_up(p.rows, Schedule::kBlockRows)
                                                          : p.rows / Schedule::kBlockRows) *
                            div_up(count, Schedule::kBlockTokens);
        if (blocks > 2147483647LL)
            throw std::invalid_argument("BF16 TMA grid exceeds CUDA grid.x capacity");
        const auto launch = [&]<bool Full>() {
            constexpr auto kernel = bf16_a16_tma_mma_kernel<Schedule, Full, Output, Epilogue>;
            constexpr int bytes =
                bf16_tma_scratch_bytes<Schedule, Epilogue> + Schedule::kBarrierBytes;
            bf16_prepare_shared<bytes, kernel>();
            kernel<<<static_cast<unsigned>(blocks), Schedule::kThreads, bytes, stream>>>(
                output, epilogue, p.rows, p.k, offset, count, p.weight, p.x);
            CUDA_CHECK(cudaGetLastError());
        };
        if (count % Schedule::kBlockTokens == 0)
            launch.template operator()<true>();
        else
            launch.template operator()<false>();
    });
}

#else

template <class Schedule, bool FullTokens, class Output, class Epilogue>
__global__
__launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void bf16_a16_tma_mma_kernel(
    const __grid_constant__ Bf16TmaDescriptors descriptors, Output output, Epilogue epilogue,
    int rows, int input_rows, int token_offset, int count) {
    constexpr int BR = Schedule::kBlockRows, BT = Schedule::kBlockTokens, BK = Schedule::kBlockK;
    constexpr int S   = Schedule::kStages;
    const int K       = Schedule::kStaticK ? Schedule::kStaticK : input_rows;
    const int tiles_r = bf16_predicated_rows<Schedule> ? (rows + BR - 1) / BR : rows / BR;
    const int tiles_t = (count + BT - 1) / BT;
    int tile_r, tile_t;
    bf16_mma_tile_coordinates<Schedule>(blockIdx.x, tiles_r, tiles_t, tile_r, tile_t);
    const int row_begin = tile_r * BR, token_begin = token_offset + tile_t * BT;
    extern __shared__ __align__(128) unsigned char storage[];
    auto* a = reinterpret_cast<__nv_bfloat16*>(storage);
    auto* b = a + S * BR * BK;
    auto* full =
        reinterpret_cast<std::uint64_t*>(storage + bf16_tma_scratch_bytes<Schedule, Epilogue>);
    auto* empty = full + S;
    if (threadIdx.x == 0) {
#pragma unroll
        for (int stage = 0; stage < S; ++stage) {
            cta_mbarrier_init(full + stage, 1);
            cta_mbarrier_init(empty + stage, Schedule::kConsumerWarps);
        }
        cta_mbarrier_fence_init();
    }
    __syncthreads();
    const int tiles_k = K / BK + (bf16_predicated_k<Schedule> && K % BK != 0);
    if (threadIdx.x < Schedule::kProducerThreads) {
        if (threadIdx.x == 0) {
            for (int kt = 0; kt < tiles_k; ++kt) {
                const int stage = kt % S;
                cta_mbarrier_wait(empty + stage, 1U ^ ((kt / S) & 1U));
                cta_mbarrier_arrive_expect_tx(full + stage, (BR + BT) * BK * 2);
                bf16_tma_load<Schedule>(a + stage * BR * BK, &descriptors.weight, kt * (BK / 64),
                                        row_begin, full + stage);
                bf16_tma_load<Schedule>(b + stage * BT * BK, &descriptors.activation,
                                        kt * (BK / 64), token_begin, full + stage);
            }
        }
        return;
    }
    const int tid  = threadIdx.x - Schedule::kProducerThreads;
    const int warp = tid / 32, lane = tid & 31;
    float accum[Schedule::kMmaRows][Schedule::kMmaTokens][4] = {};
#pragma unroll Schedule::kConsumerKUnroll
    for (int kt = 0; kt < tiles_k; ++kt) {
        const int stage = kt % S;
        cta_mbarrier_wait(full + stage, (kt / S) & 1U);
        bf16_mma_compute_stage<Schedule>(a + stage * BR * BK, b + stage * BT * BK, accum, warp,
                                         lane);
        // Every lane must finish its shared reads before the elected lane releases this stage.
        __syncwarp();
        if (lane == 0) cta_mbarrier_arrive(empty + stage);
    }
    bf16_finish_mma_tile<Schedule, FullTokens>(output, epilogue, storage, accum, row_begin,
                                               token_begin, rows, token_offset + count, warp, lane);
}

template <class Schedule, class Output, class Epilogue>
void launch_bf16_a16_tma_mma(const Bf16A16Operands& p, Output output, Epilogue epilogue,
                             cudaStream_t stream) {
    // Descriptors are launch-owned values, copied into kernel parameters during Graph capture.
    const auto descriptors = make_bf16_tma_descriptors<Schedule>(p);
    for_each_token_slice(p.tokens, Schedule::kBlockTokens, [&](int offset, int count) {
        const auto blocks = static_cast<std::int64_t>(bf16_predicated_rows<Schedule>
                                                          ? div_up(p.rows, Schedule::kBlockRows)
                                                          : p.rows / Schedule::kBlockRows) *
                            div_up(count, Schedule::kBlockTokens);
        if (blocks > 2147483647LL)
            throw std::invalid_argument("BF16 TMA grid exceeds CUDA grid.x capacity");
        const auto launch = [&]<bool Full>() {
            constexpr auto kernel = bf16_a16_tma_mma_kernel<Schedule, Full, Output, Epilogue>;
            constexpr int bytes =
                bf16_tma_scratch_bytes<Schedule, Epilogue> + Schedule::kBarrierBytes;
            bf16_prepare_shared<bytes, kernel>();
            kernel<<<static_cast<unsigned>(blocks), Schedule::kThreads, bytes, stream>>>(
                descriptors, output, epilogue, p.rows, p.k, offset, count);
            CUDA_CHECK(cudaGetLastError());
        };
        if (count % Schedule::kBlockTokens == 0)
            launch.template operator()<true>();
        else
            launch.template operator()<false>();
    });
}

#endif // NINFER_ARCH_SM89

} // namespace ninfer::ops::detail
