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
| _(baseline run in progress)_ | | | | | |

Manual reference at `93045a4b` (hand run): story generation **52 tok/s decode**, 91k prompt
**3700 tok/s prefill**. A 32-token cold `generate` at the same commit: 27.6-27.8 tok/s decode,
37.9-39.1 tok/s prefill, TTFT ~17 s - the short run carries the cold expert windows, the long
runs reach the steady state; both are comparisons against themselves, not against the hand run.

## Optimizations

_The series baseline is being measured; entries land as they earn their rows._
