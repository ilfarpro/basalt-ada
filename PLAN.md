# Plan

The current plan of scheduled work. Read it before picking up a task; update it as plans change.

## Ada (sm_89) main card - scoped to Qwen3.8-Flash-Next, UD-Q4_K_XL

The docs already name sm_89 a supported main card; the build and the runtime do not yet run it.
The scope of this section is one model and one pack class: a UD-Q4_K_XL pack of
Qwen3.8-Flash-Next running end to end on one Ada main card at a stock driver, no slower than
Strata on the same card and the same pack.  The other models (inkling2, hy4flash, deepseek41,
qwen35) and the NVFP4 packs stay out until this one ships; the narrow_bf16 work done for NVFP4
remains valid but sits off the Q4_K_XL path, and the inkling2/hy4flash read-ahead conversion was
finished just before this scope was drawn (its notes stay below as evidence).  A probe build
(`-DCMAKE_CUDA_ARCHITECTURES=89 -DBASALT_MODELS=qwen4exp`, worktree with the arch gate opened)
is the inventory below; its only compile blockers were `gdn_window.cu`, `dense_ops.cu` and
`hc_fused.cu`, and the FP4/FP8 prefill bodies.

**Bar: designed for Ada, not translated from Blackwell.** A Blackwell-only mechanism gets a native
sm_89 design - the cp.async pipelines tuned to Ada's SMs and shared memory, FP8 tensor cores used
where they are Ada's fast path, BF16 MMA where the numerics need it - not an emulation, a temporary
dequant or a lazy port of code that merely compiles. A replacement is not done when it runs; it is
done when its microbench against the measured hardware limit (GB/s, TFLOP/s) and the end-to-end
number show the card is well used. Shortcuts are rejected: a path that runs but wastes the hardware
is reworked, not shipped.

- [x] `narrow_bf16`: an sm_89 body for the E2M1/E4M3 to bf16x2 path (`cvt.*.e2m1x2` and
      `mul.rn.bf16x2` are Blackwell/sm_90+; keep the conversion exact); `narrow_bf16_parity`
      runs on sm_89.  Needed by the NVFP4 packs, not by a Q4_K_XL one: done, off this path.
- [x] Decode: `window_gemm` on sm_89 - the loads were already `cp.async` (this file has no TMA)
      and the bf16 MMA with FP32 accumulation ports as is; the one Blackwell-only instruction was
      `mul.rn.bf16x2`, whose bits sm_89 rebuilds exactly from f32 products and one
      `cvt.rn.bf16x2.f32`.  On the 4080 SUPER `window_gemm_parity` is bitwise against the double
      reference, ptxas spills nothing at 59-166 registers, and the bench sits at the card's
      measured tile-pattern ceiling (640-679 GB/s against 647), so no shape was re-tuned from
      sm_120.  Deferred with the free Blackwell card: `window_gemm_two_gpu` cross-card bitwise;
      and the test binary exits at its last check there because its reference `rms_norm_rows`
      (`dense_ops`, not in the peer file list) has no sm_89 cubin yet - a build-gate item below.
- [x] TMA audit: a `cp.async` body wherever the main path uses `cp.async.bulk` (`dec_moe`,
      `attn_chain`, `hc_fused`, the prefill GEMMs);
      thread-block clusters only through the SPLIT instantiation pattern, re-tuned for Ada's
      SM count.  Done: `dense_tc` and `qsa_mid` - the mbarrier phases stay, warp 0's bulk copy
      becomes 16-byte `cp.async.cg` chunks spread over the block with one
      `cp.async.mbarrier.arrive.noinc` per thread per issue (the plain form pre-increments the
      pending arrival count, so a phase whose every arrival is such an arrive never completes on
      sm_89 - hung on a 4080 until fixed), and the consumer spins on `test_wait.parity`, whose
      "no longer in a phase of this parity" answer is exact when every phase is awaited in turn,
      as these loops do; Ada lacks `try_wait`.  Both parity tests are bitwise on the 4080 SUPER.
      Done: `verify_kernels` - the lazy L2 read-ahead's bulk prefetch becomes 128-byte
      `prefetch.global.L2` lines walked by the same lead thread (fire-and-forget, the step's
      last-word read still waits for the part to land); `dec_commit_parity` is bitwise on the
      4080 SUPER incl. the lazy read-ahead, whose bench reads 47 MB at ~670 GB/s once the
      still-in-flight tail is discounted (the card's measured ceiling).
      Done: inkling2/hy4flash's ops and glue - their five bulk prefetches become the shared
      `basalt/kernels/l2_pf.cuh` helper (sm_90+ keeps the bulk asm verbatim, below it walks
      128-byte `prefetch.global.L2` lines; the callers still wait through their last-word
      reads).  On the 4080 SUPER all five run: inkling2's `attnres_mix`, `mix_norm` with the
      read-ahead block switched on (outputs bitwise against the run without it),
      `expert_read_ahead`, `expert_prefetch` and bitwise `moe_finish`; hy4flash's
      `engram_apply` and `l2_read_ahead`, which still stops at its mark and reads only the
      prefix.  (Out of the narrowed scope, done before it: the notes above stand as evidence.)
      Remaining, now as the chain-fallback item below: `dec_moe` already carries a designed
      non-TMA kernel path for older targets ("the host must not choose kind 1 there") - it needs
      its Ada measurement against the bar, not new code.
- [ ] Chain fallbacks made whole for an Ada main build: the Blackwell chain kernels
      (`mid_chain`'s `k_mid`, `attn_chain`, `hc_fused`, `gdn_window`'s `dec_gdn_rec` family,
      `dense_ops`' `this_cluster()` site) are capability-gated - `mid_chain_init` already logs
      "the unfused kernels run" - but the 89 probe shows the tree does not yet compile there:
      `hc_fused.cu` needs its TMA helpers behind the same `>= 900` guard `mid_chain.cu` uses;
      `gdn_window.cu`'s host side launches the cluster-only `rec_kernel`/`rec_seg_kernel`/
      `gates_kernel` with no older-target definition and `record_gdn.cpp` calls `dec_gdn_rec`
      unconditionally - GDN record needs an Ada design (the bar: measured, not just running);
      `dense_ops.cu`'s cluster form of `gdn_gating_proj_kernel` (DSMEM split reduction) needs
      its non-cluster form or gate.
      The unfused pieces every fallback names (`fused_gdn`, `gdn`, `hc8_gr`, `qsa_decode_attn`,
      `qsa_dec_fused`, the prompt `gemm`/`kernels`/`moe_fused*`, `moe_mmq`, every other
      qwen4exp kernel file) compile clean for sm_89 already.
- [ ] Build: `cmake/target_arch.cmake` takes 89 (the probe's whole-tree blockers are exactly the
      file set of the item above plus the FP4 bodies); the `120a`-only targets split into a
      portable and an arch-accelerated body - for this scope `prefill_fp4`/`prefill_fp8`/
      `prefill_mxfp8` (block-scaled `mma.sync.kind::mxf4nvf4`/`mxf8f6f4` and `cvt.e2m1x2`, hard
      ptxas rejects on 89) leave the 89 build with the packs that need them, `window_gemm` and
      `narrow_bf16_parity` already carry 89 bodies; CI builds the sm_89 main path.
      (`prefill/q8_gemm.cu`, TMA + `try_wait`, belongs to inkling2's prompt path: out of scope.)
- [ ] Runtime: `gpu_main_supported` and the card gates accept sm_89, and the NVFP4 packs are
      refused on an Ada main card with a clear message while their kernels stay unbuilt.
- [ ] Re-measure the card's gates for Ada's SM count: the QSA paths ship gated at `sms >= 128`
      (a 4090), and a 4080-class card has 80 SMs; re-tune or re-gate from measurements, not by
      lowering the number.
- [ ] First milestone: a UD-Q4_K_XL Flash-Next pack runs end to end on a 4080-class card.
- [ ] Benchmarks: Basalt vs Strata on one card, one pack, one drafter; ABBA with the power limit
      stated; decode in ms per window at T=1/4/8/16 with the acceptance rate; stock drivers only.

## Later Ada work (out of the scope above)

- [ ] NVFP4 experts on Ada: decode through the `narrow_bf16` path (done), prefill through an Ada
      design that replaces the `kind::mxf4nvf4` paths - FP8 tensor cores (Ada's full-rate path;
      a numerics move, the quality gate decides) or BF16 to FP32 - not an FP4 emulation; each
      measured before it replaces anything.
- [ ] Ada main card for the other models: inkling2's `attn.cu` calls `this_cluster()` (its
      `attn_chain`-style redesign), hy4flash's tests need `dense_ops`/`mhc` cubins; their other
      files compile clean for sm_89 already.
