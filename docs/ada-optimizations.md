# Ada (sm_89) optimizations and benchmark series

The working record of making Qwen3.8-Flash-Next fast on an Ada main card ([PLAN.md](../PLAN.md)
holds the plan; this file holds the numbers). Every optimization that claims a win records its
before/after here; a claimed win without a row here did not happen.

## The machine

GeForce RTX 4080 SUPER, 80 SMs, 16 GiB, power limit 320 W (stock), driver 615.71.09, CUDA 12.x
nvcc, DDR5 188 GiB host. The second card (RTX 5060 Ti) runs a serve and is never benchmarked on.

## The harness

- **Series:** `basalt-bench --community --model /mnt/2TB/Basalt/models/Qwen3.8-FN-UD-Q4_K_XL.basalt
  --submitter ada-main --gpu 0 -- -kv int8 -expert-cache 2300` run from `bin/` of an sm_89 build.
  Cold process per fixture (conv, struct, prose decode; 16k, 64k prefill), int8 KV, the pack's
  drafter. `--expert-cache 2300` because the auto fill leaves no margin for the verify window
  graphs on this card.
- **Manual reference (the user's runs):** a story-about-the-moon generation, and a 91k-token
  prompt prefill.

| date | HEAD | decode conv/struct/prose (tok/s) | decode weighted | prefill 16k / 64k (tok/s) | notes |
|---|---|---|---|---|---|
| 2026-11-26 | 93045a4b | 37.1 / 52.6 / 38.0 | 48.3 | 2675 / 2947 | baseline; accept 70.9/98.1/67.7%, draft len 1.7/6.9/1.6; ~231 W of 320 drawn at ~2600 MHz SM while decoding |

Manual reference at `93045a4b` (hand run): story generation **52 tok/s decode**, 91k prompt
**3700 tok/s prefill**. A 32-token cold `generate` at the same commit: 27.6-27.8 tok/s decode,
37.9-39.1 tok/s prefill, TTFT ~17 s - the short run carries the cold expert windows, the long
runs reach the steady state; both are comparisons against themselves, not against the hand run.

## What the SM-count gates actually hold on this card (2026-11-26, at 93045a4b)

The "re-tune the `sms >= 128` gates" plan item resolves into one live gate and one dead:

- **`hc_fused` / `mid_chain` are not SM-gated, they are architecture-gated.** Both kernels' bodies are
  `#if __CUDA_ARCH__ >= 900` - TMA bulk copies and mbarriers, which Ada sm_89 does not have (only
  Ampere cp.async). `mid_chain_init` reports "not covered (80 SMs, 101376 B optin smem; need 16,
  101376)" but the two numbers it prints both pass; the failing condition is its
  `hc_fused_supported()` guard. Ada runs the unfused hc8 kernels and the mid layer runs as staged
  (`cp.async`), checked bit for bit - no re-tune reaches these. The warning text should name the
  reason it actually refuses (tracked below).
- **`qsa_qproj_half` is genuinely Ada-capable and the one live SM gate.** `qsa_mid.cu`/`dense_tc.cu`
  carry `>= 800` and a `>= 800 && < 900` Ada branch; the fused half path is a performance policy, and
  the only term of the gate that fails at 80 SMs is `sms >= 128`. At this geometry the other two pass
  with room: `(12288+15)/16 = 769 > 80` tiles (the multi-tile layout), and the halves' tiles are
  `2*16*(2560/256)*32 = 10240 B` against a 90 KiB bound. So enabling it on this card is one
  measurement, not a port.

## Optimization: enable `qsa_qproj_half` at 80 SMs (the live SM-gate)

`qsa_qproj_half` splits the Q-projection GEMV (N=2560 to NH*2*HD=12288) into a query half and a gate
half: the query half runs on block-per-SM (80 blocks), the gate half on 3/8 of the SMs (30 blocks)
beside the indexer-key branch, off the attention's critical path. Measured on `qsa_mid_parity --bench`
(halves cold from many weight copies past L2; µs per call, the halves' combined output is the same
bytes as dense_tc's) and end-to-end with the community preset:

| arm | result |
|---|---|
| _measurement in progress_ | |

Design note: the gate's `sms >= 128` was tuned on a Blackwell card. On 80 SMs the two halves launch
80 + 30 blocks, which is already a 1.4x oversubscription before the attention and the indexer join,
so whether the gate-half's overlap still pays is exactly what has to be measured. The candidate
retune is to drop the `sms >= 128` term (the other two terms already encode the shape and smem
conditions) or restate it against the tile count, then verify by benchmark.
