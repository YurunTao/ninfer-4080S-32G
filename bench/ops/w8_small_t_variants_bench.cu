// Internal schedule sweep for the W8 small-T MMA LinearSwiGLU route on sm_89.
//
// The registered small-T tables (W8SmallTMmaDefaultSchedule and the per-geometry production
// tables) were tuned on the RTX 5090. This bench instantiates candidate W8SmallTMmaSchedule
// variants directly - bypassing the route table - and times them cold-cache at the decode
// token widths of the Qwen3.8-27B gate/up projection, so an Ada retune can be chosen from
// measurements instead of inherited constants. Bench-only: nothing here changes product
// dispatch.
//
// Usage: ninfer_w8_small_t_variants_bench [--t-sweep 4,8,16] [--warmup N] [--repeat N]
//                                         [--csv-out PATH]

#include "ninfer_bench_common.h"
#include "quantized_weight.cuh"

#include "core/device.h"
#include "ops/common/memory.cuh"
#include "ops/linear/w8/w8_small_t_mma.cuh"
#include "ops/linear_swiglu/w8/w8_linear_swiglu_output.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace ninfer;
using namespace ninfer::ops;
using namespace ninfer::ops::detail;

namespace {

constexpr std::int32_t kIntermediate = 17408;  // Qwen3.8-27B gate/up output rows
constexpr std::int32_t kHidden       = 5120;
constexpr std::size_t kFlushBytes    = 256ULL << 20;

using Launcher = void (*)(const __nv_bfloat16*, const std::uint8_t*, const std::uint8_t*,
                          __nv_bfloat16*, cudaStream_t);

template <int ActiveCols, int KWarps, int MinBlocks, W8SmallTMmaScaleAccess Scale,
          W8SmallTMmaActivationStage Stage>
void launch_variant(const __nv_bfloat16* x, const std::uint8_t* codes, const std::uint8_t* scales,
                    __nv_bfloat16* out, cudaStream_t stream) {
    constexpr int kTileTokens =
        ActiveCols <= 8 ? 8 : (ActiveCols <= 16 ? 16 : (ActiveCols <= 24 ? 24 : 32));
    using Geometry  = W8LinearGeometry<2 * kIntermediate, kHidden>;
    using RowPolicy = W8SwiGluPairedRows<kIntermediate>;
    using Schedule =
        W8SmallTMmaSchedule<KWarps, kTileTokens, MinBlocks, Scale, Cache::ca, Cache::cg, Stage>;
    const W8ContiguousOutput ignored_output{out, kIntermediate};
    const W8SwiGluDirectEpilogue epilogue{out, kIntermediate};
    w8_small_t_mma_kernel<Geometry, ActiveCols, Schedule, W8ContiguousOutput,
                          W8SwiGluDirectEpilogue, RowPolicy, true>
        <<<kIntermediate / RowPolicy::kOutputRowsPerCta, Schedule::kThreads, 0, stream>>>(
            x, codes, scales, ignored_output, epilogue, RowPolicy{});
}

struct Variant {
    std::int32_t active_cols;
    std::string name;
    Launcher launcher;
};

template <int ActiveCols, int KWarps, int MinBlocks, W8SmallTMmaScaleAccess Scale,
          W8SmallTMmaActivationStage Stage>
void add(std::vector<Variant>& out) {
    const char* scale_name = Scale == W8SmallTMmaScaleAccess::Direct ? "direct" : "shared";
    const char* stage_name =
        Stage == W8SmallTMmaActivationStage::PaddedZero ? "pad0" : "active";
    char name[96];
    std::snprintf(name, sizeof(name), "w=%d mb=%d scale=%s stage=%s", KWarps, MinBlocks,
                  scale_name, stage_name);
    out.push_back(
        {ActiveCols, name, &launch_variant<ActiveCols, KWarps, MinBlocks, Scale, Stage>});
}

template <int ActiveCols, int KWarps, int MinBlocks>
void add_scale_stage(std::vector<Variant>& out) {
    // A launch bound that cannot fit the SM thread limit is not a real candidate.
    static_assert(KWarps * 32 * MinBlocks <= 1536, "min blocks per SM exceeds the thread limit");
    add<ActiveCols, KWarps, MinBlocks, W8SmallTMmaScaleAccess::Shared,
        W8SmallTMmaActivationStage::ActiveOnly>(out);
    add<ActiveCols, KWarps, MinBlocks, W8SmallTMmaScaleAccess::Direct,
        W8SmallTMmaActivationStage::ActiveOnly>(out);
    if constexpr (ActiveCols <= 16) {
        add<ActiveCols, KWarps, MinBlocks, W8SmallTMmaScaleAccess::Shared,
            W8SmallTMmaActivationStage::PaddedZero>(out);
    }
}

template <int ActiveCols, int KWarps>
void add_warps(std::vector<Variant>& out) {
    add_scale_stage<ActiveCols, KWarps, 1>(out);
    add_scale_stage<ActiveCols, KWarps, 2>(out);
    if constexpr (KWarps * 32 * 3 <= 1536) { add_scale_stage<ActiveCols, KWarps, 3>(out); }
    if constexpr (KWarps * 32 * 5 <= 1536) { add_scale_stage<ActiveCols, KWarps, 5>(out); }
}

template <int ActiveCols>
void add_width(std::vector<Variant>& out) {
    add_warps<ActiveCols, 4>(out);
    add_warps<ActiveCols, 8>(out);
    // Sixteen warps only fit the 48 KiB static staging at the eight-token tile.
    if constexpr (ActiveCols <= 8) { add_warps<ActiveCols, 16>(out); }
}

std::vector<Variant> make_variants() {
    std::vector<Variant> variants;
    add_width<4>(variants);
    add_width<8>(variants);
    add_width<16>(variants);
    return variants;
}

std::vector<std::int32_t> parse_t_sweep(std::string_view raw) {
    std::vector<std::int32_t> result;
    std::size_t begin = 0;
    while (begin < raw.size()) {
        const std::size_t end = raw.find(',', begin);
        const std::string token(
            raw.substr(begin, end == std::string_view::npos ? raw.size() - begin : end - begin));
        const long value = std::stol(token);
        if (value != 4 && value != 8 && value != 16) {
            throw std::invalid_argument("--t-sweep values must be 4, 8, or 16");
        }
        result.push_back(static_cast<std::int32_t>(value));
        if (end == std::string_view::npos) { break; }
        begin = end + 1;
    }
    if (result.empty()) { throw std::invalid_argument("--t-sweep must not be empty"); }
    return result;
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::vector<std::int32_t> t_sweep{4, 8, 16};
        int warmup = 5;
        int repeat = 30;
        std::string csv_out;
        for (int i = 1; i < argc; ++i) {
            const std::string_view arg(argv[i]);
            const auto next = [&](const char* label) -> std::string_view {
                if (++i >= argc) { throw std::invalid_argument(std::string("missing ") + label); }
                return argv[i];
            };
            if (arg == "--t-sweep") {
                t_sweep = parse_t_sweep(next("--t-sweep value"));
            } else if (arg == "--warmup") {
                warmup = std::stoi(std::string(next("--warmup value")));
            } else if (arg == "--repeat") {
                repeat = std::stoi(std::string(next("--repeat value")));
            } else if (arg == "--csv-out") {
                csv_out = next("--csv-out path");
            } else if (arg == "--help" || arg == "-h") {
                std::printf("Usage: %s [--t-sweep 4,8,16] [--warmup N] [--repeat N] "
                            "[--csv-out PATH]\n",
                            argv[0]);
                return 0;
            } else {
                throw std::invalid_argument("unknown argument: " + std::string(arg));
            }
        }

        DeviceContext context;
        const cudaStream_t stream = context.stream;
        std::printf("# gpu=%s sm=%d%d cuda_runtime=%d cache=cold\n", context.props.name,
                    context.props.major, context.props.minor, CUDART_VERSION);
        DeviceBuffer flush(kFlushBytes);
        DeviceBuffer input(static_cast<std::size_t>(kHidden) * 16);
        DeviceBuffer output(static_cast<std::size_t>(kIntermediate) * 16 * 2);
        auto packed = bench::make_row_split_weight(QType::W8G32_F16S, 2 * kIntermediate, kHidden,
                                                   kHidden, {0x31, 0x00, 0x3c00});
        const double weight_bytes = static_cast<double>(packed.model_weight_bytes());
        const auto variants       = make_variants();
        std::printf("# variants=%zu weight_bytes=%.0f\n", variants.size(), weight_bytes);

        for (const std::int32_t t : t_sweep) {
            Tensor x(input.p, DType::BF16, {kHidden, t});
            Tensor out(output.p, DType::BF16, {kIntermediate, t});
            std::vector<std::pair<double, const Variant*>> ranking;
            for (const Variant& variant : variants) {
                if (variant.active_cols != t) { continue; }
                const auto body = [&](cudaStream_t launch_stream) {
                    variant.launcher(static_cast<const __nv_bfloat16*>(x.data),
                                     static_cast<const std::uint8_t*>(packed.weight.qdata),
                                     static_cast<const std::uint8_t*>(packed.weight.scales),
                                     static_cast<__nv_bfloat16*>(out.data), launch_stream);
                };
                const bench::ColdTiming timing =
                    bench::measure_cold_launch(body, flush, stream, warmup, repeat);
                const double bytes = weight_bytes + 2.0 * kHidden * t;
                const double gbps  = bytes / timing.median_us / 1e3;
                ranking.push_back({gbps, &variant});
            }
            std::sort(ranking.begin(), ranking.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
            std::printf("== T=%d ==\n", t);
            for (const auto& [gbps, variant] : ranking) {
                std::printf("  %-46s %8.1f GB/s\n", variant->name.c_str(), gbps);
            }
            if (!csv_out.empty()) {
                std::string path = csv_out;
                if (t_sweep.size() > 1) { path += ".T" + std::to_string(t); }
                std::FILE* file = std::fopen(path.c_str(), "w");
                if (file == nullptr) { throw std::runtime_error("cannot open CSV"); }
                std::fprintf(file, "T,variant,logical_gbps\n");
                for (const auto& [gbps, variant] : ranking) {
                    std::fprintf(file, "%d,%s,%.1f\n", t, variant->name.c_str(), gbps);
                }
                std::fclose(file);
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_w8_small_t_variants_bench: %s\n", error.what());
        return 1;
    }
}
