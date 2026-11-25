# Ada (sm_89) optimizations and benchmark series

The working record of making Qwen3.8-Flash-Next fast on an Ada main card ([PLAN.md](../PLAN.md)
holds the plan; this file holds the numbers). Every optimization that claims a win records its
before/after here; a claimed win without a row here did not happen.

## The machine

GeForce RTX 4080 SUPER, 80 SMs, 16 GiB, power limit 320 W (stock), driver 615.71.09, CUDA 12.x
nvcc, DDR5 188 GiB host. The second card (RTX 5060 Ti) runs a serve and is never benchmarked on.

## The harness

- **Series:** `basalt-bench --community --model /mnt/2TB/Basalt/models/Qwen3.8-FN-UD-Q4_K_XL.basalt
  --submitter ada-main --community-dir DIR` (the preset takes the engine flags itself: `--gpu 0
  --max-context 66661 --kv int8`, expert cache auto). Cold process per fixture (conv, struct, prose
  decode; 16k, 64k prefill), no warm-up, no repetition - every number is a cold one.
- **A/B harness for experiments:** `basalt-bench -t conv,struct,prose -r 2 --seed 7 --gpu 0
  -o json --logs DIR -- --kv int8 [the one flag under test]` - one warm engine, two repetitions,
  a fixed sampling seed so both arms sample the same text; judged on ms per decode window, not
  tok/s. int8 KV because that is what the serve runs and what arms the decode path with the fused
  QSA prelude; the fp16 default runs a different engine and is not comparable.
- **Manual reference (the user's runs):** a story-about-the-moon generation, and a 91k-token
  prompt prefill.

| date | HEAD | decode conv/struct/prose (tok/s) | decode weighted | prefill 16k / 64k (tok/s) | notes |
|---|---|---|---|---|---|
| 2026-11-26 | 93045a4b | 37.1 / 52.6 / 38.0 | 48.3 | 2675 / 2947 | baseline; accept 70.9/98.1/67.7%, draft len 1.7/6.9/1.6; ~231 W of 320 drawn at ~2600 MHz SM while decoding |

Flagged: the same build at the warm A/B harness measured 57.5 / 90.6 / 61.2 tok/s for the three
fixtures - the preset's own cold numbers sit 35-53% below. The preset starts each fixture from a
process on an idle GPU (210 MHz idle clocks, graphs uncaptured, the profile prefill of 9.5 GB into
the cache); the suspected cause is the clock ramp and graph capture inside a ~25 s measurement.
Series entries stay comparable to each other; do not compare them to the warm harness row.

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

## Is decode RAM-bound on this machine? (2026-11-26, the user's question)

The machine's steady state includes the user's serve (NVFP4 pack on the 5060 Ti): it pins 73.5 GiB
of experts in RAM and spins ~10 cores (986% CPU) - my bench pins another 67.6 GiB, together about
141 of the 188 GiB, and both engines' expert pools read the same DDR5 channels. The hand benchmark
runs in the same conditions, so the comparison holds.

- Expert traffic, conv fixture: ~597 expert lookups per generated token, cache hit 62.9% at
  3040 VRAM slots of 24576, the average blob 2.9 MB: about 640 MB of RAM reads per token. At the
  measured 57.5 tok/s this is ~37 GB/s for one engine; the harness's RAM read rate measures
  49.6-56.6 GB/s at DDR5-4000. One engine is at 65-75% of the ceiling, and the serve's pool wants
  its share on top: the channels are effectively saturated.
- The CPU expert pool scales +21% from 8 to 19 workers on conv, then only +2% from 19 to 24 with
  cores left idle - the classic memory-bound flattening, not a compute one.
- The user's own experiment agrees from the other side: DDR5-4000 to 5600 gave +17% decode
  (63.3 -> 74) with prefill flat (2638 -> 2663) on the 5060 Ti.

**Verdict: decode here is RAM-bandwidth-bound; prefill is not.** DDR5-5600 (theoretical +40%)
should give roughly +10-20% on low-acceptance decode (conv, prose, stories), little on
high-acceptance (struct), ~0 on prefill. Two flags for the DIMM decision: the capacity math -
two engines want 141 GiB of pinned RAM, which will not fit two sticks' capacity, so with 5600
MT/s run one engine at a time; and `sudo dmidecode -t memory` may show the four current sticks
are single-rank and could hold a higher speed without removal (a board/BIOS question, not a
physics one). The `--pool-workers` knob above the auto size changes decode by only a couple of
percent here; the bandwidth is the wall.

## Optimization: `qsa_qproj_half` at 80 SMs - measured, the gate stands

`qsa_qproj_half` splits the Q-projection GEMV (N=2560 to NH*2*HD=12288) into a query half and a gate
half: the query half runs on block-per-SM (80 blocks), the gate half on 3/8 of the SMs (30 blocks)
beside the indexer-key branch, off the attention's critical path. The kernel's Ada branch had never
run on this card; with the gate lowered past `sms >= 128` for the measurement:

- `qsa_mid_parity --bench` on the 4080: **bitwise against dense_tc for all seven formats** (T 1..8),
  Q6_K/Q8_0 timed at T 1..4: dense_tc 41.6/50.8 us, the halves alone 22.6+26.9 / 26.2+27.2 us -
  the two halves together move the same bytes as dense_tc but need ~19% more time on 110 blocks of
  an 80-SM card; whether the hidden gate-half still wins is an end-to-end question.
- End to end, the only configuration where the path can run (int8 KV makes `fused_pre` true; the
  fp16 runs do not reach it): conv, two repetitions per arm, ms per decode window 40.748/40.796
  with the path ON vs 40.776/40.809 OFF, tok/s 57.6/59.0 vs 57.5/59.0 - the difference is noise.
  At T=1 the Q-projection's share of the 20.5 ms window is what the halves could shorten, and the
  halves' own sum is slower than the whole they replace.

**Verdict: the `sms >= 128` term is right for this card too, confirmed at 80 SMs - no change.**

## Optimization: expert cache and pool sizing on the 4080 (measuring)

_measuring: pool workers 28/32, expert cache beyond the auto 3040 slots_
