# NInfer-4080S-32G

NInfer-4080S-32G 在单张 **RTX 4080 SUPER 32G**（AD103，80 个 SM，32760 MiB 显存，
736 GB/s 显存带宽）上运行 **Qwen3.8-27B**，提供完整的 262,144 token 原生上下文。
它是 [NInfer-4090](https://github.com/sergiuszm/ninfer-4090)（`sm_89` 分支）的
RTX 4080 SUPER 32G 移植，后者是
[NInfer-3090](https://github.com/Don-Chad/ninfer-3090)（`sm_86`）的移植，最终源自
[Neroued/ninfer](https://github.com/Neroued/ninfer) —— 一个从零实现的 C++20/CUDA
单 GPU 推理引擎，面向显式注册的 `.ninfer` 工件做极限单卡性能优化。

引擎加载官方 groupwise `.ninfer` 工件，提供 OpenAI 与 Anthropic 兼容 API，支持 paged KV、
兼容前缀复用、CUDA Graphs、MTP 投机解码、DFlash2 投机解码
（`--spec dflash2 --draft-tokens 7`，需经工件转换器加入伴随权重）、推理力度控制、
ReplaySSM 状态事务。Blackwell 专属的 NVFP4/W4A4 执行不可用；Qwen3.6-35B-A3B 目标
保留文本 DFlash（本卡未测）。

这张 32 GB 的卡带来一个实际收益：**INT8 KV（最高精度档位）可以直接跑满 262K 全上下文**，
不必像 24 GB 的 4090 那样用 4-bit E8 量化去换上下文容量。E8 各档位
（`rk8v4`/`rk4v4`/`rk4v4-e8`/`rk2v4-e8`）仍然可用，用于需要更多显存余量或更高并发的场景。

## 仓库结构

| 分支 | 内容 |
|---|---|
| `master` | 上游 [Neroued/ninfer](https://github.com/Neroued/ninfer) master 的快照（sm_120a / RTX 5090 主线，完整历史） |
| `4080s-32g` | 本分支：RTX 4080 SUPER 32G 移植（快照式历史：4090 fork 基线 + 4 个提交） |

本分支的历史是快照式的（基线提交包含全部引擎源码），与 `master` 的上游历史没有共同
祖先；跨分支对比以内容为准，不以提交图为准。

## 本机实测（RTX 4080 SUPER 32G）

条件：单请求、贪心解码、CUDA Graphs 开启、INT8 KV、代码生成语料；驱动 610.47、
CUDA 13.0、`sm_89`。bench 数值取自 `ninfer_bench`（`profiles/bench/retune4080s/`），
服务侧数值与 bench 的 INT8 A/B 一致。

| 测试 | 结果 |
|---|---|
| Prefill 512 | 1338 tok/s |
| Prefill 2048 | 1371 tok/s（`--prefill-chunk 2048` 为最优点，见下） |
| Decode 128，无投机 | 37.15 tok/s |
| Prefill 2048 + Decode 128，DFlash2 K=7 | prefill 1343 / **decode 156.66 tok/s** |
| DFlash2 K=7 接受率 | 88.0%，接受长度 7.11 / 8（逐位置 94, 94, 94, 94, 83, 83, 67 %） |

DFlash2 解码是普通解码的 **4.29 倍**。解码是权重带宽受限：每轮搬运约 17.6 GiB 权重，
实际约 500 GB/s（相对卡的 736 GB/s 峰值）；T=8 宽度下 W8 gate/up 投影的算子上限约
540 GB/s，端到端解码达到该上限的 93%。

### 运行时旋钮扫描（免重编译）

| 旋钮 | 结果 |
|---|---|
| `--prefill-chunk` 512 / 1024 / 2048 / 4096 | 1277 / 1342 / **1365** / 1314 tok/s —— 取 2048 |
| DFlash2 `--draft-tokens` K=4…9 | 123.1 / 130.4 / 145.0 / **158.9** / 133.8 / 137.3 tok/s —— 取 K=7 |
| K=7 + `--lm-head-draft`（优化提案头） | 160.5 tok/s，接受率 88 % |
| K=7 + `--no-cuda-graph` | 157.5 tok/s（CUDA Graphs 约 +2 %） |
| K=7，不启用 `--lm-head-draft`（草稿自带提案头） | **175.6 tok/s**，接受率 100 %，接受长度 8.0 |

`--lm-head-draft` 与草稿自带提案头在本卡的代码语料上出现反转：后者更高。叙事类语料
的结论不同（见下），实际配置按工作负载选择。

### W8 小 T 调度重调（c01–c05，ABBA 成对测量）

上游的 W8 小 T MMA 表是在 170-SM 的 RTX 5090 上调的。本分支用成对 A/B 方法
（基线与候选二进制交替运行，抵消时钟/热漂移）逐一验证：

| id | 候选 | 解码 Δ（DFlash2） | 解码 Δ（无投机） | 结论 |
|---|---|---|---|---|
| c01 | T≤4：16 → 8 warp | -5.7 % | -1.2 % | 不采用 |
| c02 | T≤4：16 → 4 warp | +0.6 % | +0.7 % | 保留 16 |
| c03 | T 5..16：8 → 4 warp | -1.5 % | -0.1 % | 保留 8 |
| c04 | 默认表（词表等）：8 → 4 warp | +0.2 % | -3.6 % | 保留 8 |
| c05 | causal `SmallTSplitScale` 1 → 2 | -0.7 % | +2.7 % | 保留 1（开放项） |

全部落在噪声区间或为负：**本分支保留 5090 调优的 W8 小 T 表与 causal 几何**。
c05 在无投机解码上有 +2.7 % 的苗头，待 W8 静态→动态 smem 改造后重新评估（见开放项）。

### 真实工作负载（服务侧实测）

DFlash2 的接受率随输出可预测性变化：结构化代码约 **72.7 %**，叙事类约 **25.7 %**。
叙事类负载下 MTP3 更好（**78.1** vs DFlash2 61.6 tok/s）；代码生成负载下 DFlash2
大幅领先（本机基准 156.7 vs 无投机 37.2 tok/s）。KV 精度/容量不影响解码时间，这是
INT8 与 E8 档位在本卡上解码侧表现一致的原因。

## 本分支做了什么（汇总）

基线：4090 fork（[sergiuszm/ninfer-4090](https://github.com/sergiuszm/ninfer-4090)，
分支 `rtx4090-port`）的完整工作，快照式引入。本分支在此之上：

- **DFlash2 投机解码 + 上游合并**（`d40a90ed`）。把 DFlash2 投机解码（启动时选择
  草稿窗口 K=1..15，`--spec dflash2 --draft-tokens K`）连同上游 a16b644285 一并移植到
  `sm_89` 分支；伴随权重经工件转换器并入 `.ninfer` 工件。
- **80-SM 设备常量重调**（`b1b891b3`）。上游把若干启动器/规划器常量在 170-SM 的
  5090 上调成字面量，本提交换成真实设备事实或 80-SM（AD103）实测预算：
  - GDN 门控路由分裂阈值（27B split8 ≤768 / split2 ≤1664；35B split8/4/2 ≤960/1920/3840），
    与 sm_89 上实测的 2/1 与 2/4/3 CTA-per-SM 预算一致；
  - GDN 协同驻留：27B split4/2 → 1 CTA/SM，35B split8/4/2 → 3 CTA/SM（原高估值会让
    驱动拒绝 cooperative 网格）；
  - GDN 分块输出与 sparse_moe prefill 的常驻网格改按真实 SM 数计算，消除 170-SM
    字面量造成的 2.1 倍常驻波超订；
  - causal 小 T 分裂上限改为每波 SM/4 个 KV head（替换 170-SM 字面量 42）；
  - rmsnorm 预取跨越与 rope 波容量改从设备 SM 数推导（顺带修好一个编译不过的
    constexpr 初始化）。

  实测：MTP3 与 DFlash2 K=7 在短、4.6K、6.6K 上下文的 prefill/decode 均在 ±0.3 %
  内，上述路由不在关键路径上。
- **bench 合并遗留修复**（`fca5b3e8`）。dflash2 合并后首次全树构建暴露出两个漏改的
  bench 源文件，本提交补齐：`gdn_layer_bench` 适配 `DeviceExecutionView` 契约与
  `Prefetch` 模板参数（取 `false` 时与合并前 kernel 位精确一致）；`kv_cache_append_bench`
  完成半迁移（paged 前缀容量常量、`empty` 判定、6 参数 `PrefixCase`、`Result` 新字段
  与报表列、cyclic 环 V 面按当前契约为 FP16）。
- **W8 小 T 调度变体基准**（`e363920f`）。`w8_small_t_variants_bench` 直接实例化
  `W8SmallTMmaSchedule` 候选（绕过路由表），冷缓存计时 Qwen3.8-27B gate/up 投影，
  一次跑完 T=4/8/16 的 warp/min-block/scale/staging 全空间。实测 T=8 宽度下 90 个
  变体全部落在 540 GB/s 的 1 % 之内——kernel 受权重访问模式限制而非受分块限制。

从 4090 fork 继承（本分支原样携带）：`sm_89` retarget 与 INT8 注意力 prefill 的 Ada
重调、causal-tile 分区 key-block 遍历、`/v1/models` 的 `context_window` 字段、
llama.cpp 兼容 `timings`、Prometheus `GET /metrics`、`GET /slots` 槽位表、slot 会话
保存/恢复（`--slot-save-path`）、复用感知的 lane 选择、轮检查点环（`--turn-checkpoints`）、
驱逐自动保存（`--auto-save-evicted`）、NVFP4-A4 测试门控、E8 格 KV 量化与
262K–1M 可见 key 包络、可配置视觉暂存区（`--vision-max-tokens`）。详见
[docs/](docs/README.md) 与 4090 fork 的 README。

### 开放项（未合入）

- W8 静态 → 动态 smem（19 个启动点）：恢复上游 8/16-warp 形状在大 token 分块上的
  可用性，之后重跑 c03/c05 一类的宽度验证；
- MTP 侧小 T 表（`W8LinearSmallTProductionSchedule<W8Mtp*>`），仅在 MTP3 成为目标
  路径时相关；
- sparse_moe `PathsPerBlock=3` 与 `SmallTMaximumSplits = 85 * SmallTSplitScale`
  （35B-A3B 路径，本工件不经过）；
- 若 prefill 将来成为瓶颈，再引入运行时 prefill 分块开关。

## 快速开始（Linux）

要求：RTX 4080 SUPER 32G（`sm_89`，80 SM）、近期 NVIDIA 驱动、CUDA 13.0、Linux。

```bash
# 构建（本机已验证的原生构建）
cmake -S . -B build-sm89 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-sm89 -j
# 或用 Docker：docker build --tag ninfer-4080s-32g:sm89 .

# 官方工件（16.95 GiB）
NINFER_MODEL_DIR="$PWD/models" bash scripts/download-qwen38.sh
```

DFlash2 工件由本仓库 `tools/reference` 的 Python 转换器生成（需要 torch 环境）：以
官方 `Qwen/Qwen3.8-27B` 权重为底，合并
[z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) 的
伴随权重，配方 `qwen3_8_27b-v2`，输出约 19.0 GiB 的 `.ninfer`。

服务（262K 全上下文、INT8 KV、DFlash2 K=7、4 路并发、视觉开启）：

```bash
./build-sm89/apps/ninfer-serve out/et27b_dflash2.ninfer \
  --host 127.0.0.1 --port 8080 \
  --max-context 262144 --kv-capacity 262144 --kv-dtype int8 \
  --spec dflash2 --draft-tokens 7 \
  --max-concurrency 4 --vision
```

API 位于 `http://127.0.0.1:8080/v1`。CLI 单请求：

```bash
./build-sm89/apps/ninfer out/et27b_dflash2.ninfer --prompt "..." \
  --max-context 262144 --kv-capacity 262144 --kv-dtype int8 \
  --spec dflash2 --draft-tokens 7 --vision
```

叙事类负载可改用 MTP3（官方工件即可，无需伴随权重）：

```bash
./build-sm89/apps/ninfer models/qwen3_8_27b.ninfer --prompt "..." \
  --kv-dtype int8 --spec mtp --draft-tokens 3
```

启动时的显存预算（262K INT8 KV，本卡实测）：DFlash2 工件 18.27 GiB 权重 + 12.12 GiB
KV + 状态池，剩余约 1.65 GiB；官方工件 16.95 GiB 权重，剩余约 3.0 GiB。服务器在
监听前校验显存，超配的上下文会快速失败而不是在请求时才报错。

## 本卡的已知限制

- `--prefill-chunk` 取 2048 或更小：本机实测 4096 掉到 1314 tok/s（低于 2048 的
  1365）。本分支携带实测的 `sm_89` cooperative 驻留表，≤2048 的分块保持 split-K
  路径。
- W8 小 T 表与 causal 几何保留 5090 值（c01–c05 无胜者）；c05 的 +2.7 %（无投机
  解码）是已知的开放优化点。
- `--lm-head-draft` 与草稿自带提案头的优劣随语料反转：代码语料下后者更高
  （175.6 vs 160.5 tok/s），叙事类下结论相反。按工作负载选择。
- 35B-A3B 目标与 Windows 路径从上游继承，但未在本卡测量；本分支的全部实测都是
  27B 目标。
- 继承的引擎限制同样适用：单进程、单 GPU、单模型、有界 FIFO 准入、无主动请求
  抢占、无多 GPU 执行、无权重卸载。
- 与 llama.cpp 的对比数据来自 4090 fork（见
  [docs/llamacpp-comparison.md](docs/llamacpp-comparison.md)），尚未在本卡复测；
  本卡的定性结论一致：prefill 略落后，解码（尤其 DFlash2/MTP 投机）领先。

## 工件

| 模型 | 工件 | 大小 |
|---|---|---:|
| Qwen3.8-27B（官方 groupwise） | [neroued/Qwen3.8-27B-NInfer](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) | 16.95 GiB |
| Qwen3.8-27B + DFlash2 伴随权重 | 经工件转换器合并 [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) 后生成 | 19.0 GiB |

工件与硬件无关；模型卡上的 RTX 5090 要求描述的是上游引擎，不是文件本身。下载后请
按模型卡公布的 SHA-256 校验。

## 推理力度

Qwen3.8-27B 有三个训练过的推理深度加一个关闭开关。OpenAI Chat Completions 接受顶层
`reasoning_effort` 字段（`low`、`medium`、`xhigh`）与顶层 `enable_thinking` 布尔值；
隐藏推理单独返回为 `message.reasoning_content`。llama.cpp 的 `chat_template_kwargs`
请求字段不受支持，会被拒绝。CLI 用 `--reasoning-effort` 或 `--no-thinking`。采样默认
值来自模型卡，并随思考模式切换。

## 服务 API

OpenAI Chat Completions、带流式与本地续写状态的 OpenAI Responses、Anthropic Messages、
带已解析工具调用的 prompt 渲染函数工具、兼容前缀复用、JSONL 请求日志。详见
[HTTP 服务](docs/serving.md) 与 [CLI 用法](docs/cli.md)。

## 上游与致谢

- [Neroued/ninfer](https://github.com/Neroued/ninfer) —— 引擎本体，面向 RTX 5090
  （`sm_120a`）开发。
- [Don-Chad/ninfer-3090](https://github.com/Don-Chad/ninfer-3090) —— SM86 兼容层、
  ReplaySSM 集成、Qwen3.8 运行时支持，是 4090 fork 的基础。
- [sergiuszm/ninfer-4090](https://github.com/sergiuszm/ninfer-4090) —— RTX 4090 fork
  （`rtx4090-port` 分支），本分支的直接基线：`sm_89` retarget、INT8 注意力 prefill
  Ada 重调、causal-tile 分区遍历、E8 格 KV、槽位持久化与检查点环等全部继承自它。
- [UDPSendToFailed/ninfer-4090](https://github.com/UDPSendToFailed/ninfer-4090) ——
  同源的 RTX 4090 移植，旋转/E8 格 KV 量化模式（`rk8v4`/`rk4v4`/`rk4v4-e8`/
  `rk2v4-e8`）、E8 编解码器与 1M 可见 key 包络出自该 fork，cherry-pick 时保留了
  作者署名；完整对比见 [docs/udp-fork-comparison.md](docs/udp-fork-comparison.md)。
- [shantanusingh16/ninfer-4090](https://github.com/shantanusingh16/ninfer-4090) ——
  llama.cpp 兼容 `timings` 字段的贡献来源。

## 支持

NInfer 是一个个人兴趣项目。如果你觉得它有用并想支持后续开发，可以在
[Ko-fi](https://ko-fi.com/neroued) 上支持这个项目。

支持完全自愿。它不是购买或投资，不附带财务回报、承诺的服务或功能，也不参与项目
决策。项目的方向、优先级、技术选型和发布节奏由维护者独立决定。

## 许可

Apache License 2.0，见 [LICENSE](LICENSE)。
