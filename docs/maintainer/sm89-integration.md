# RTX 4080 SUPER (sm_89) integration — verification record

Branch: `4080s-32g` / `master` at the commit that adds this file.
Target: RTX 4080 SUPER 32G (AD103, 80 SM), CUDA 13.2, VS 2026, `CMAKE_CUDA_ARCHITECTURES=89`.

## What this branch contains

Upstream `Neroued/ninfer` master (81c8ce09, 256 commits past the true fork point
6b94b8c5) merged onto this branch together with the maintained `sergiuszm/rtx4090-port`
line (Ada/sm_89 work: INT8 group-64 activation prefill for the Q4/Q5 body, gating
residency per launched variant, the native Windows build, portable 128-bit arithmetic,
the per-item vision budget fix, tolerant Qwen tool-call recovery).

The sm_90+/sm_120a-only kernels upstream added are replaced by sm_89 routes rather than
removed:

| Upstream route | sm_89 route in this branch |
|---|---|
| bf16 A16 TMA MMA | same tile/pipeline/MMA schedule, tiles staged by `cp.async` commit groups, `__syncthreads()` orders stage reuse (`src/ops/linear/bf16/bf16_a16_tma_mma.cuh`) |
| fp8 A8 TMA MMA | same treatment, plain row-major E4M3 tiles, paired row policy kept (`src/ops/linear/fp8/fp8_a8_tma_mma.cuh`) |
| block-scaled FP8 MMA (`kind::mxf8f6f4`) | plain `mma.sync.m16n8k32.e4m3.e4m3.f32`; the block-scale form used unit scales, so the product is identical (`src/ops/common/mma.cuh`) |
| NVFP4 A4 (block-scale MMA + TMA) | not compiled; the entry points reject explicitly (`src/ops/nvfp4_sm89_stubs.cpp`, `nvfp4_attn_input_a{4,16}.cu`) |
| `cvt.bf16x2.e4m3x2` / `e2m1x2` pair widening | the exact FP16 bridge and the FP32 expansion respectively (`fp8_a16_codec.cuh`, `nvfp4_codec.cuh`) |
| programmatic dependent launch | stream-ordered launches, as the sm_86 port already did (`src/core/pdl.cuh`) |

`NINFER_ARCH_SM89` (published to every internal target by `cmake/NinferTargets.cmake`)
selects these routes, so the same sources also still build for the upstream target.

## Verification on the RTX 4080 SUPER

Artifact: Qwen3.8-27B groupwise-int, converted from v2 with
`tools/upgrade_ninfer_v2_to_v3.py` (18.2 GB, weights preserved).

```
ninfer.exe <artifact> --prompt <2088-token prompt> --max-context 32768 \
    --max-new 128 --prefill-chunk 2048 --kv-dtype int8 --greedy
```

| Measurement | This branch | 2026-09-15 baseline |
|---|---|---|
| prefill (2088 tokens, chunk 2048) | **1.37k tok/s** | 1371 tok/s @ 2048 |
| decode, no speculation | **39.2 tok/s** | 37.15 tok/s |
| decode, MTP draft 3 | **83.6 tok/s** | — |
| decode, MTP draft 1 / 5 | 63.1 / 75.9 tok/s | — |
| engine ready | 13.9–14.2 s (weights 15.9 GiB at 4.42 GiB/s) | — |
| output correctness | correct instruction following, greedy | — |

INT8 group-64 KV (`--kv-dtype int8`) runs at the same decode rate as BF16 KV, and the
engine loads, plans and generates end to end from a clean build.

## Not measured here

`--spec dflash2` validation is complete (it reports `missing component dflash2` for an
artifact without companion draft weights) but its throughput is not measured on this
branch: the DFlash2 companion weights come from `z-lab/Qwen3.8-27B-DFlash2` and the
previously converted artifact is v2. Regenerating it needs that checkpoint plus
`tools/convert`, so the 2026-09-15 DFlash2 figure (156.66 tok/s) has no same-branch
counterpart yet. MTP above exercises the same speculative round machinery.

## Known gaps

- Retained-session slot persistence (`--slot-save-path`, `save_slot`/`restore_slot`/
  `erase_slot`/`session_index_size`) rejects explicitly: the store belongs to the
  maintainer line's engine core, and this branch runs upstream's engine core and model
  layer, which do not carry it. See `src/runtime/engine/session_slots.cpp`.
- `tests/` is not built in this configuration (`BUILD_TESTING=OFF`); the test tree still
  contains files written against the maintainer line's artifact reader API.
