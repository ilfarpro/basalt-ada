# Ada (sm_89) optimizations and benchmark series

The working record of making Qwen3.8-Flash-Next fast on an Ada main card ([PLAN.md](../PLAN.md)
holds the plan; this file holds the numbers). Every optimization that claims a win records its
before/after here; a claimed win without a row here did not happen.

## The machine

GeForce RTX 4080 SUPER, 80 SMs, 16 GiB, power limit 320 W (stock), driver 615.71.09, CUDA 12.x
nvcc, DDR5 188 GiB host. **Memory speed: every row before 2026-11-26 08:00 ran at DDR5-4000;
the owner then ran the box at 5600 MT/s** (root would be needed for `dmidecode` to re-read it;
the speed is the owner's report). The serve's home moved with the swap: through the night the
second card (RTX 5060 Ti) ran the user's serve and was never benchmarked on; since 08:53 the
user's serve (the same UD-Q4_K_XL pack, ctx 204800, this repo's 07:01 engine binary) runs on
the 4080 SUPER itself and the 5060 Ti is idle - from now on the main card's own serve is the
confounder to account for in every series.

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
- **ctest hygiene (06:55):** on this PC `ctest` must run with `CUDA_VISIBLE_DEVICES=0`. A
  CVD-less run fails 9 tests (gdn_chunk_parity, prompt_split_state_test, dec_moe_parity_kq,
  cvec_parity, host_registration_test, window_gemm_two_gpu, expert_tier_test, ds41_peer_parity,
  peer_share_parity) environmentally - they touch the other card or two-device paths and fight
  the live serve. Under CVD=0 the full suite lands **230/231 (99%)**: the only failure is
  `platform_cpu_test` (pre-existing, unrelated to this work) and the two-GPU tests
  (`window_gemm_two_gpu`, `peer_share_parity`) skip correctly on a single visible card
  (full-suite run 06:52, log `/tmp/ctest_final_cvd.log`, test tree `/tmp/build-89c` matching
  the sm_89 build at 11ef05b9).
- **Manual reference (the user's runs):** a story-about-the-moon generation, and a 91k-token
  prompt prefill. The user's method (verified 07:35): the serve from the moon section (GPU 0,
  port 18090, ctx 204800, defaults - no temperature manipulation); after the load first a
  short "Привет" turn, then a long Russian moon prompt ("long story about the moon backed by
  real scientific facts, own knowledge, no tools"); the rate is read over a **long**
  generation - 800 tokens is too short to see the stable ceiling. On this build (07:01, HEAD
  3651cfb0) the user measures **55 tok/s decode (52 on the 93045a4b-era build) with the
  prefill unchanged at ~3700** - accepted as the interim Ada-main result.
- **Moon reproduction (corrected 06:30, corrected again 07:15):** the one-shot form is
  `bin/basalt --model <pack> --tokens "<ids>" --max-new 800 --kv int8 --vram-reserve-mib 128`
  (there is no `generate` subcommand to pass positionally before the flags - `basalt generate
  ...` is refused as `unknown argument: generate`). moon_ids = the tokens of
  `"Write a short story about the moon."` wrapped in the pack's chat template (16 tokens:
  `248045,846,198,7734,264,2716,3255,883,279,17257,13,248046,198,248045,74455,198`, encoded
  with the repo's own `tools/tokenizer.py`). The one-shot path needs a low cache cap
  to leave graph headroom (see the knife-edge section: `--expert-cache 1200`); with the
  chat-template ids it decodes 800 moon tokens at **45.3 tok/s** (accept 67.5%, 1.83 tokens
  per round - an earlier 86.3 figure came from a raw-ids prompt and is retracted below);
  the user-like serve path (a second serve on GPU 0, ctx 16384, one chat call with
  `max_tokens: 800`) measured 41.8 tok/s end-to-end cold, 66.0 hot (the engine's own tg lines:
  51.3 cold / 66.2 hot, hit 64.5-72.4%) - the user's hand figure of
  52 tok/s sits between (their serve: ctx 200k elastic under live traffic). Texts differ run
  to run (temperature 0.7); the row compares build to build, not the story.

| date | HEAD | decode conv/struct/prose (tok/s) | decode weighted | prefill 16k / 64k (tok/s) | notes |
|---|---|---|---|---|---|
| 2026-11-26 | 93045a4b | 37.1 / 52.6 / 38.0 | 48.3 | 2675 / 2947 | **superseded by the clean row below** - taken before serve CPU-seconds were accounted and the serve had live traffic through it; accept 70.9/98.1/67.7%, draft len 1.7/6.9/1.6; ~231 W of 320 drawn at ~2600 MHz SM while decoding |
| 2026-11-26 04:30 | 09941828 | 57.4 / 89.8 / 61.4 | 80.4 (preset's own) | 3520 / 3758 | **the clean HEAD series**: same preset, same pack, serve 0 CPU-s during the whole 210 s run (verified); accept 69.1/98.1/68.0%, draft len 1.8/6.9/1.5 - the workload is identical to the 93045a4b row, the code differs only in the log warning, so the old row's -35% is contamination (the serve's ~20-core bursts), not the cold preset: these cold-process numbers match the warm A/B harness's steady state (57.5-59.0 / 90.6-91.3 / 61.2-62.2). Page cache hot (pack mapped in 15 s); ram_gbps recorded 56.8 |

Resolved (04:30): the clean cold-process series above sits at the warm harness's steady state,
so the preset is not intrinsically slow - the baseline row sat 35-53% low because the serve's
~20-core bursts ran through it (it predates the per-arm serve accounting). The idle-clock ramp
theory does not survive: with the serve quiet, a cold process on freshly-captured graphs measures
the same as the warm harness. Every row from tonight carries its serve CPU-seconds. Rows dated
2026-11-26 before 08:00 are DDR5-4000 territory (see The machine; the 5600 series is due).

Manual reference at `93045a4b` (hand run): story generation **52 tok/s decode**, 91k prompt
**3700 tok/s prefill**. The owner re-verified the same way at the 07:01 build (HEAD `3651cfb0`,
their method: "Привет" warm-up turn, long Russian moon story, default sampling, rate read over
the long generation): **55 tok/s decode, prefill ~3700 unchanged** - accepted as the interim
Ada-main result. Re-measured on DDR5-5600 at the same build and method (08:47): **decode 67
tok/s, prefill 4068** - the swap row is under the RAM verdict below. A 32-token cold
`generate` at the same commit: 27.6-27.8 tok/s decode,
37.9-39.1 tok/s prefill, TTFT ~17 s - the short run carries the cold expert windows, the long
runs reach the steady state; both are comparisons against themselves, not against the hand run.

## What the SM-count gates actually hold on this card (2026-11-26, at 93045a4b)

The "re-tune the `sms >= 128` gates" plan item resolves into one live gate and one dead:

- **`hc_fused` / `mid_chain` are not SM-gated, they are architecture-gated.** Both kernels' bodies are
  `#if __CUDA_ARCH__ >= 900` - TMA bulk copies and mbarriers, which Ada sm_89 does not have (only
  Ampere cp.async). `mid_chain_init` reports "not covered (80 SMs, 101376 B optin smem; need 16,
  101376)" but the two numbers it prints both pass; the failing condition is its
  `hc_fused_supported()` guard. Ada runs the unfused hc8 kernels and the mid layer runs as staged
  (`cp.async`), checked bit for bit - no re-tune reaches these. The warning now names the term
  that refused (11ef05b9).
- **`qsa_qproj_half` is genuinely Ada-capable and the one live SM gate.** `qsa_mid.cu`/`dense_tc.cu`
  carry `>= 800` and a `>= 800 && < 900` Ada branch; the fused half path is a performance policy, and
  the only term of the gate that fails at 80 SMs is `sms >= 128`. At this geometry the other two pass
  with room: `(12288+15)/16 = 769 > 80` tiles (the multi-tile layout), and the halves' tiles are
  `2*16*(2560/256)*32 = 10240 B` against a 90 KiB bound. So enabling it on this card is one
  measurement, not a port.

## Is decode RAM-bound on this machine? (2026-11-26, the user's question)

The machine's steady state includes the user's serve (NVFP4 pack on the 5060 Ti): it pins 73.5 GiB
of experts in RAM, its host loop spins a core, and its real traffic arrives in bursts that hold
10-20 cores for minutes at a time - my bench pins another 67.6 GiB, together about 141 of the
188 GiB, and both engines' expert pools read the same DDR5 channels. The hand benchmark
runs in the same conditions, so the comparison holds.

- Expert traffic, conv fixture: ~597 expert lookups per generated token, cache hit 62.9% at
  3040 VRAM slots of 24576, the average blob 2.9 MB; the dispatcher computes a window's distinct
  experts once, so ~460-640 MB of RAM reads per token. At the measured 57.5 tok/s that is 26-37
  GB/s for one engine; the harness's RAM read rate measures 49.6-56.6 GB/s at DDR5-4000 - 50-75%
  of the ceiling with the serve's pool wanting its share on top.
- The CPU expert pool scales +21% from 8 to 19 workers on conv, then only +2% from 19 to 24;
  28 workers (exactly the hardware-thread count) collapsed to 28.8 tok/s - SMT siblings thrash the
  L1/L2 the AVX-2 kernels stream through. The pool is near its ceiling; more cores do not help.
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

**Swap measured on the main card (08:47, the owner's run).** DDR5-5600, the same 07:01 engine
binary that measured 55 / ~3700 at DDR5-4000, the same serve command (the user's serve now on
the 4080 SUPER itself), the same method ("Привет" warm-up, long Russian moon story, default
sampling, rate over the long generation; the 91k-prompt prefill): **decode 55 -> 67 tok/s
(+21.8%), prefill 3700 -> 4068 tok/s (+9.1%)**. Decode landed just above the predicted
+10-20% band - the verdict holds. Prefill beating its ~0 prediction has a clean explanation:
the +0 was measured on the 5060 Ti serve, which runs `--resident-experts` (its prompt experts
never touch RAM); this serve streams the prompt path's experts from RAM (the night's runs read
~57 GB/s through the pool), so its prefill rides the channels too and takes ~9% of the +40%
theory. With 67 tok/s at 5600 the earlier per-knob measurements remain closed (the knobs moved
with the wall, not against it), but the community series and the harness's ram_gbps predate the
swap - re-run both before quoting a baseline again.

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

## dec_moe's designed non-TMA path against the card's limits (`dec_moe_parity --kq --bench`, 02:3x)

The last PLAN bar item measured on the 4080 SUPER, all bitwise:

- resident VRAM grouped call, T=8, 28 groups, 73.8 MB: 108.65 us = 680 GB/s, and its glue alone
  87.14 us = 847 GB/s - at or above the card's measured tile-pattern ceiling (647 GB/s).
- the staged PCIe copy (the streamed-experts path) runs 17.1-19.9 GB/s at 8/16/32/64 blocks
  against the probed link's 18.4 - the link is the limit and the kernel reaches it; staged
  compute of 7 groups is 22.7-25.4 us.

Nothing sits between these kernels and their limits: closed.

## The one-shot VRAM knife-edge (diagnosed 06:35) and what the moon run revealed

The same one-shot command that completed cleanly at 01:1x began failing at
`verify: instantiate: out of memory` at 06:12. Root cause, measured with caps (the ~1-3 GiB
driver-side loss first suspected is retracted - there was none): the expert-cache auto-sizer
grabs essentially all remaining VRAM ("9.94 GiB free" -> a 9.55 GiB cache) and the verify
window (106 MiB), the MTP draft graphs, the PDL capture pool and cuBLAS workspaces then share
~0.3 GiB with the auto-borrowed prompt-path slots - a knife-edge that a few hundred MiB of
driver-side drift (idle `Reserved` read 350 MiB tonight) tips either way. Two facts make this
path brittle: the auto-sizer reserves for "what loads after" but not for the graph pools, and
**one-shot prompt-path borrowing adds ~330-371 slots on top of any `--expert-cache` cap**
(asked 1200 -> 1529 actual, asked 3000 -> 3371; the serve path's cap is exact - 3179 -> 3179
verified) - so caps of 2600-3100 still starved the graphs and looked like lost VRAM. With a
cap low enough to leave graph headroom (`--expert-cache 1200`) the run completes and
story-about-the-moon decodes (default small ctx; TTFT excludes init). The serve path never
hits the knife-edge (elastic KV yields cells to the cache and back): measured 41.8 cold /
66.0 hot tok/s end-to-end at ctx 16384 with the same pack and build.

The 86.30 read from that path is **retracted** (07:15). That run passed a raw 16-token id list
(`1207,318,8261,...` - a pre-tokenization from a foreign vocab, no chat template), and what it
generated drafted at **99.8% accept, 2.92 tokens per round** - the real chat-template moon
prompt runs 59-73% accept, 1.7-2.1 tokens per round. The same one-shot with the chat-template
ids and the same 1200 cap decodes the 800-token story at **45.32 tok/s**, right in the serve
path's cold-to-hot band. The "working set fits the cache" reading was wrong twice over: that
run swapped 3909 experts into the tier mid-decode (constant churn, no fit), and 1529 cache
slots did not make it faster than the serve's 2939. What the numbers say instead: the verify
round rate is near-constant at 25-34 rounds/s across one-shot and serve, 1529 and 2939 slots,
ctx 16k and 200k - and decode tok/s = round rate x tokens per round, so the 45-86 spread
between moon runs is draft acceptance (text difficulty), not the cache. Comparing moon rows
means quoting accept and tokens-per-round next to tok/s.

The serve slot-trade follow-up (smaller `--max-context` or kv-resident returns slots to the
expert cache) is **measured, and it is not a lever** (07:04, serve on GPU 0, quiet window,
0 serve CPU-s): KV is elastic - the ctx-204800 serve holds it in 0.7/3.1 GiB of cuda0 and gets
the same **2939-slot** cache as the ctx-16384 one (cuda0 14.0 vs 14.3 GiB); hot moon tg is
66.1 vs 66.2 tok/s, hit 68.9 vs 72.4%. Context trades nothing. The user's 52 tok/s sits inside
this regime (accept ~59-73%, conversations cold-to-warm); the engine-side lever for moon
remains the round rate itself - RAM bandwidth (the DDR5 verdict) and CPU-pool speed - not
slot counts.

## Basalt vs Strata on the same card and pack (blocked on RAM, found 05:31; dropped by the owner 07:49 - kept as the measured record, do not restage)

Strata 0.2.21.1 is installed with the same pack (`Qwen3.8-Flash-Next-UD-Q4_K_XL`, GGUF shards).
The saved run configs behind the profiles were stale (they point at control/0.1.63, deleted by
the update; installed is 0.1.68), so an isolated copy was made (`/tmp/strata_cfg_q4_4080.json`:
paths 0.1.63 -> 0.1.68, GPU 0 pinned via `CUDA_VISIBLE_DEVICES`, port 18080, log in /tmp) and a
new profile `ada-night-compare` created from it - the user's own profiles and configs untouched.
The engine came up (its PCIe probe 18.2 GB/s -> `pcie_frac 0.50`, the same adaptive rule Basalt
computes; KV streaming 32768 of 204800 cells in VRAM) and two seconds into mapping its 72 GiB
pinned complement **systemd-oomd killed the scope** (kernel: "Cannot map memory with base addr
... size of 0x11eedcf pages"; oomd's top candidate was the scope at 89.8 GiB, memory pressure
Avg10 89.6). That is the machine saying no: the serve already pins 73.5 GiB and a second ~72 GiB
engine plus its client does not fit - one engine under that pinning is fine, two are not. The
comparison needs the serve stopped (the user's call, it carries their traffic) or more RAM; it
cannot be had passively overnight. Everything for it is staged: the `standard-v1` suite (short
decode, 4k/16k prefill, concurrency ladder c1-c4 x r1-r3, warmup) is one quiet
`strata bench run --profile ada-night-compare --suite standard-v1` away. Drafters differ by
default (Strata runs `--spec 4 --spec-min-p 0.5`, Basalt the engine default 8) and the protocols
differ (community-1 is cold per fixture, standard-v1 is served state): the comparison will be
per-engine steady states, guarded by an ABBA repetition that guards drift, not protocol. The
320 W power limit gets stated with the row. Decision (07:49): the owner dropped the comparison
and the serve-stopped window; what the section keeps is the machine fact - this RAM holds one
engine under ~72 GiB pinning, two never.

## The expert cache hit rate and the learned profile (measured: cold-start value only)

The user's serve (read-only log, `/home/ilfarpro/.cache/basalt/aefd5d98dc650ed4/engine.log`,
03:07): decode expert cache hit rate 51.6-60.8% per request window on the bundled static profile,
3304 slots (elastic: shrinks to 3284 when KV needs cells), 279M blobs served from RAM since start,
zero from file. The engine can learn its own ranking (`--expert-profile-save`: heat accumulates
through the session, written at stdin close/QUIT and on a timer; a present file replaces the
model's profile at the next serve start - `load/model.cpp`, `serve/tier_profile.cpp`).

Measured (03:49/03:58, clean paired warm arms, serve 0 and 1 CPU-s): conv 41.59/44.49 ms/window
(OFF) against 40.20/44.46 (ON), struct and prose within noise; per-request hit rates 61-67.5% in
both arms. That is the whole effect and it is small, and the reason is structural: the adaptive
tier already re-places the cache toward live traffic within minutes, so a learned file only
buys the first minutes after a start - on a 14-hour session both arms reach the same placement.
The serve's steady 52-61% is therefore the post-adaptation ceiling for its slot count, not a
deficit a profile can fix; raising it means more VRAM slots, which the card does not have
(the `--expert-cache` cap experiment, above, moved the hit rate zero). Verdict: worth switching
on for the serve (it is free, quality-neutral, and warms the first minutes after every restart),
worth no expectation of steady-state speed. The profile file `/tmp/lp/all.prof` from
conv+struct+prose is the artifact.

## Knobs measured and rejected (2026-11-26)

- **Expert cache beyond the auto 3040 slots**: `--expert-cache 3179` is what actually fits on this
  card (auto sits at 96% of that). Paired arms: hit 63.3/62.9% vs 62.5/64.8%, windows 40.0/43.2 vs
  35.1/42.9 ms - inside each other's spread, and 4.6% more slots cannot move a 37% miss rate. Dead.
- **Pool workers**: auto 19 is right; 24 buys ~2% (noise-grade), 8 loses 19%, 28 (the thread count)
  collapses to 28.8 tok/s on SMT thrash. Do not ship a bigger default for hybrid CPUs on evidence
  this thin, and do not let users set the thread count as a "tune".
- **Spec depth**: `--spec` floors at 6, and the pack's drafter drafts 6.9 tokens on struct -
  depth 6 costs struct 12-23% (draft_len caps at 3.0) while conv gains nothing. Default 8 stands.
- **`qsa_qproj_half`**: closed above. **PCIe share**: `pcie_frac` is not a Blackwell constant, it is
  computed at start from the probed link (0.55 x min(1, GB/s / 20); this card: 17.3 -> 0.50) -
  already Ada-adaptive, nothing to retune.
- **`--prefill-cpu-share`**: 0.3 against the default auto (0.6), clean paired prefill arms
  (hot page cache, serve 0 CPU-s both): 3508.8/3779.2 against 3510.9/3781.8 tok/s at 16k/64k -
  identical. The flag does nothing measurable on this 20-core machine; default stands.
  The pair also calibrates the page cache: **hot-cache prefill is ~31% above the cold community
  row** (3510/3780 against 2675/2947) - the user's hand figure (3700 tok/s at 91k) is hot-cache
  territory, since a hand run repeats over a pack the RAM already holds. The community row stays
  the cold canonical; night chains see hot caches after their first load and must not be
  cross-compared to it.
