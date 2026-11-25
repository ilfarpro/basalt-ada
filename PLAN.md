# Plan

The current plan of scheduled work. Read it before picking up a task; update it as plans change.

## Ada (sm_89) main card

The docs already name sm_89 a supported main card; the build and the runtime do not yet run it.
The goal: Qwen3.8-Flash-Next on Ada at a stock driver, no slower than Strata on the same card and
the same K-quant pack.

**Bar: designed for Ada, not translated from Blackwell.** A Blackwell-only mechanism gets a native
sm_89 design - the cp.async pipelines tuned to Ada's SMs and shared memory, FP8 tensor cores used
where they are Ada's fast path, BF16 MMA where the numerics need it - not an emulation, a temporary
dequant or a lazy port of code that merely compiles. A replacement is not done when it runs; it is
done when its microbench against the measured hardware limit (GB/s, TFLOP/s) and the end-to-end
number show the card is well used. Shortcuts are rejected: a path that runs but wastes the hardware
is reworked, not shipped.

- [ ] Build: `cmake/target_arch.cmake` takes 89; the `120a`-only targets (`window_gemm`,
      `prefill_fp4`/`prefill_fp8`/`prefill_q8`, `narrow_bf16_parity`) split into a portable body
      and an arch-accelerated one; CI builds the sm_89 main path.
- [ ] Runtime: `gpu_main_supported` and the card gates accept sm_89.
- [x] `narrow_bf16`: an sm_89 body for the E2M1/E4M3 to bf16x2 path (`cvt.*.e2m1x2` and
      `mul.rn.bf16x2` are Blackwell/sm_90+; keep the conversion exact); `narrow_bf16_parity`
      runs on sm_89.
- [x] Decode: `window_gemm` on sm_89 - the loads were already `cp.async` (this file has no TMA)
      and the bf16 MMA with FP32 accumulation ports as is; the one Blackwell-only instruction was
      `mul.rn.bf16x2`, whose bits sm_89 rebuilds exactly from f32 products and one
      `cvt.rn.bf16x2.f32`.  On the 4080 SUPER `window_gemm_parity` is bitwise against the double
      reference, ptxas spills nothing at 59-166 registers, and the bench sits at the card's
      measured tile-pattern ceiling (640-679 GB/s against 647), so no shape was re-tuned from
      sm_120.  Deferred with the free Blackwell card: `window_gemm_two_gpu` cross-card bitwise;
      and the test binary exits at its last check there because its reference `rms_norm_rows`
      (`dense_ops`, not in the peer file list) has no sm_89 cubin yet - a build-gate item below.
- [ ] TMA audit: a `cp.async` body wherever the main path uses `cp.async.bulk` (`dec_moe`,
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
      prefix.  The models' full parity binaries do not run on Ada yet: inkling2's `attn.cu`
      calls `this_cluster()` (the `attn_chain` item) and hy4flash's needs `dense_ops`/`mhc`
      cubins (the `hc_fused` item); the models' other files compile clean for sm_89 and
      wait for those items.
- [ ] Prefill: replace the `kind::mxf4nvf4` paths with an Ada design, not an FP4 emulation: an
      FP8 tensor-core GEMM (Ada's full-rate path; a numerics move, the quality gate decides) or
      BF16 to FP32.
- [ ] First milestone: a UD-Q4_K_XL Flash-Next pack runs end to end on a 4080-class card.
- [ ] NVFP4 experts on Ada come last: decode through the `narrow_bf16` path, prefill through the
      Ada prefill path, each measured before it replaces anything.
- [ ] Benchmarks: Basalt vs Strata on one card, one pack, one drafter; ABBA with the power limit
      stated; decode in ms per window at T=1/4/8/16 with the acceptance rate; stock drivers only.
