# Plan

The current plan of scheduled work. Read it before picking up a task; update it as plans change.

## Ada (sm_89) main card - scoped to Qwen3.8-Flash-Next, UD-Q4_K_XL

The docs already name sm_89 a supported main card; the build and the runtime now run it (the items
below are done and measured, the 4080 SUPER numbers in docs/ada-optimizations.md).
The scope of this section is one model and one pack class: a UD-Q4_K_XL pack of
Qwen3.8-Flash-Next running end to end on one Ada main card at a stock driver. (The
no-slower-than-Strata bar: the face-off was dropped by the owner on 2026-11-26 - the owner's
own runs, recorded under Benchmarks below, are the acceptance.)  The other models (inkling2, hy4flash, deepseek41,
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
      Done: the dec_moe non-TMA path measured against the bar on the 4080 SUPER
      (`dec_moe_parity --kq --bench`, 2026-11-26): the resident VRAM grouped call reads
      73.8 MB at 680 GB/s (108.65 us) and its glue at 847 - at or above the card's measured
      tile-pattern ceiling (647), and the staged PCIe copy runs 17-20 GB/s at 8/16/32/64
      blocks, the probed link's own 18.4.  Nothing between the kernel and its limit there.
- [x] Chain fallbacks made whole for an Ada main build: `hc_fused.cu`'s TMA body is sm_90-and-up
      guarded and the host runs the unfused `hc8_gr` there (3a830b07); `dense_ops.cu`'s gating
      GEMM gained an older-target whole-K copy beside the untouched Blackwell text, measured on
      a 4080 SUPER in `dense_ops_parity`: 88.5 us against the unfused 116.2 (3be8c171);
      `gdn_window.cu`'s fused entries run the split family on an older target's build (the gates
      inside the kernel or in window-sized pieces through `dec_gdn_gates`, the raw output in its
      scratch, the gated norm after it, and the split kernel writes `state_out` there),
      `dec_gdn_rec_seg` carrying the window row count for it; `dec_gdn_parity` (cluster gate
      dropped, a bitwise state_out-vs-commit check added) and `gdn_seg_split_parity` pass
      bitwise on the 4080 SUPER, and `mid_chain`'s `k_mid`/`attn_chain` stay capability-gated
      at run time (a2a77a47: the sm_120 SASS of `gdn_window`, `mid_chain` and `dec_moe` is
      unchanged, the peer TU's object byte-identical).
      The rest (`fused_gdn`, `gdn`, `qsa_decode_attn`, `qsa_dec_fused`, the prompt
      `gemm`/`kernels`/`moe_fused*`, `moe_mmq`, every other qwen4exp kernel file) already
      compiles clean for sm_89.
- [x] Build: `cmake/target_arch.cmake` takes 89 (one compute capability per build, its own message), and an
      89 build defaults to the one model an Ada main card serves (`cmake/models.cmake`); the
      `120a`-only prefill bodies - `prefill_fp4`/`prefill_fp8`/`prefill_mxfp8` (`mma.sync.kind::mxf4nvf4`/
      `mxf8f6f4`, `cvt.e2m1x2`) and `prefill_q8` (TMA + `try_wait`, another model's prompt path) - stay in
      the 89 build as portable sections: the byte-count helpers stay live (scratch sizing needs them),
      every launch entry refuses with a named message (the runtime refuses those packs before them, the
      next item); the guarded regions are HEAD's text (their sm_120a SASS is byte-identical);
      `hc_fused.cu`'s Blackwell helpers dead-code on 89 gets a file- and arch-scoped `--diag-suppress=177`
      instead of the Werror it would hit; the whole tree builds for 89 with `WERROR=ON` and its CPU tests
      pass; CI gains `build-ada-main` (compile + CPU tests, qwen4exp). `gpu_select.cpp`'s sm_89-as-peer
      special cases now mean "sm_89 as a foreign arch" (they no longer fire when 89 is the build's own
      capability) and `gpu_select_test` names the build's capability through `kCudaCCMajor/Minor`.
- [x] Runtime: the card gates accept sm_89 (`gpu_main_supported` follows `cuda_cc_is_ours`, the build item
      above), and a native pack with NVFP4 experts is refused at the start of an older target's build, in the
      format check `qfn_run.cpp` already runs before any allocation ("its NVFP4 kernels are not built") -
      its kernels stay unbuilt there.
- [x] The suite runs clean on an 89 build on the 4080 SUPER, the machine's own CPU-topology test excepted
      (a pre-existing `platform_cpu_test` anomaly on this kernel): the older target's recorded test answers
      skip (26f8109a), `hc_fused_parity` and `dec_moe_parity_*` register the 77 their gates already returned,
      and the tag protocol's host producers are clean. A GeForce RTX 4080 SUPER evicts a dirty CPU cache line
      when it pleases and its late write-back overwrote what a combine had re-armed into a peer's row (the tag
      word lost in most late-landing rows; no store qualifier, fence or verify-and-retry changed it - the
      line simply survived with no pressure, and NT stores leave no line to evict): `fill_u32_nt`/`copy_f32_nt`
      (`compiler.hpp`) write every host word of the tag rows and the window's word block, and
      `dec_moe_parity --kq` passes ten runs in a row.
- [x] Two suite flakes under `-j`: `dec_gdn_parity`'s `state_out` poison ran on the legacy stream its
      non-blocking test stream does not wait for, so the poison could land after the kernel that had committed
      its state (a bitwise DIFF only under load); it waits now. `expert_cache_segmented_test` and
      `kv_arena_test` measure the device's free VRAM, a total a GPU test running beside them moves mid-check:
      they run serially now, and their readings retry inside a short grace as signed deltas.
- [x] Re-measure the card's gates for Ada's SM count: the QSA paths ship gated at `sms >= 128`
      (a 4090), and a 4080-class card has 80 SMs; re-tune or re-gate from measurements, not by
      lowering the number.  Done (docs/ada-optimizations.md): `hc_fused`/`mid_chain` are not SM-gated
      but architecture-gated (TMA bodies, `major == 12`), unreachable by retune; `qsa_qproj_half`'s
      gate is the one live SM term and was measured with it lowered - bitwise on the Ada branch,
      and zero at the window on the path's only live configuration (int8 KV), the halves being 19%
      slower per byte at kernel level on 110 blocks of an 80-SM card.  The `sms >= 128` term stands,
      validated by measurement.  The decode pool, expert cache and spec-depth knobs were measured
      too: all at their ceilings on this machine, which is RAM-bandwidth-bound for decode.
- [x] First milestone: a UD-Q4_K_XL Flash-Next pack runs end to end on a 4080-class card.
      (Done on a 4080 SUPER: prefill 39 tok/s, decode 27.8 tok/s over 32 tokens with spec/MTP; the card's
      expert cache needs a --expert-cache margin for the window graphs - the auto fill left none, and the
      verify graph instantiate failed.)
- [x] Benchmarks: the Basalt rows are banked; the repeating harness is `basalt-bench` (the
      `--community` preset: three cold decode fixtures plus 16k/64k prefills) and its Ada series
      with every optimization's before/after lives in `docs/ada-optimizations.md`. The owner
      verified the interim result by hand (their method: "Привет" warm-up turn, then a long
      Russian moon story, default sampling, rates read over long generations - 800 tokens is
      too short to see the ceiling): **55 tok/s decode (52 on the 93045a4b-era build)** with
      the 91k-prompt prefill unchanged at **~3700 tok/s** - accepted as the interim Ada-main
      result. The owner then swapped the host to DDR5-5600 and re-ran the same build the same
      way: **decode 67, prefill 4068** (+21.8% / +9.1%) - the RAM-bandwidth read-out, in
      `docs/ada-optimizations.md`; the harness series (at 4000) is due a re-run on the new
      speed. The Basalt-vs-Strata face-off on this card is **dropped by the owner
      (2026-11-26 07:49)**: the two engines' ~72 GiB pinned complements cannot coexist in this
      RAM (systemd-oomd kills the second loader; the measured record stands in
      `docs/ada-optimizations.md`) and a serve-stopped window was declined. Not to be restaged.

## Ada main card + Blackwell (sm_120) peer - the mirrored cross-capability build

`-DPEER_BLACKWELL=ON` is the mirror of `-DPEER_ADA=ON`: an sm_89 build compiles the peer-tier
kernel list (the `dec_moe` peer variants, the peer's MMQ instances, the iq/quantize/wait/sampler/
elementwise helpers, plus `native_moe.cu` as `dec_moe_parity`'s reference side on the peer card)
also for sm_120, so a Blackwell card can hold and compute experts for an Ada main card. The gates
accept a foreign sm_120 peer only under the flag (`gpu_peer_supported`, the `choose_gpus` refusal
text, `check_gpu_modes` through the new `GpuModes::peer_blackwell`), and Flash-Next is the model
whose peer kernels are in that set. The peer runs only the shared standard-CUDA kernels, so no
new kernel bodies and no kernel source changed.

- [x] Build option and guards (needs an 89 build, refuses together with `PEER_ADA`), the
      mirrored source list, gates and refusal texts, `GpuModes::peer_blackwell`, the
      `gpu_select_test` cases for both flag states and the model gate; the full-suite build
      with `-DWERROR=ON`.
- [x] `window_gemm_two_gpu` skips (77) when the second card does not run this build's kernels:
      it replans main-card kernels there, which is the split rule of the same capability, never
      the cross-capability peer. Its cross-card bitwise item below "Later Ada work" stays open.
- [x] Docs: "an older peer card" becomes "a peer card of the other capability" in
      `docs/configuration/gpus.md` (both directions, speeds not ages); `docs/platforms.md`
      names both flags.
- [x] Two-card GPU verification on the 4080 SUPER + 5060 Ti pair, in a serve-stopped window (the
      main card is fully held while the serve runs). Passed 8/8: `dec_moe_parity`'s cross-card
      bitwise gate (the peer kernel against `native_expert_grouped`, both on the Blackwell card),
      `peer_share_parity`'s peer MMQ share against the main card and the double CPU reference,
      `expert_tier_test`, `ds41_peer_parity`, `host_registration_test`. The first run also caught
      three suite tests that replay main-card kernels on the second device (`prompt_split_state_test`,
      `gdn_chunk_parity`, `cvec_parity`) plus `peer_share_parity`'s own swiglu on the peer stream:
      a card of the other capability runs none of those kernels and fell through to the PTX JIT
      (this machine's driver is one CUDA minor under the toolkit, which rejects its PTX). Fixed:
      the three skip such a device the way `window_gemm_two_gpu` does, the test file gets the
      peer's SASS under a cross-capability peer flag; the rerun passes 8/8.
- [ ] End to end: `--gpu 0,1 --gpu-mode peer` on the UD-Q4_K_XL pack, decode and prefill against
      the one-card baseline (67 / 4068 tok/s at DDR5-5600), recorded in `docs/ada-optimizations.md`
      before any speed claim. This machine's peer slot is electrically PCIe 4.0 x4 (~6.5-7 GB/s):
      it moves the expert load at start and any row crossing, not the steady-state decode math,
      which keeps expert rows on the card that holds them.
      First serve start caught a cross-capability e2e bug the unit tests missed: with a peer card
      the engine auto-enables the prompt split (whole layers on the peer), the verifier head split,
      and the GDN mixer split, all of which run the *model's own* kernels (attention, dense GEMM,
      `q80p_vhead`) on that card - compiled only for the main capability, so the peer card threw them
      into the PTX JIT. A cross-capability peer build compiles only the peer *tier*'s kernels for the
      peer card, not the model's layers, so the peer card is the expert tier and nothing else. Fixed:
      `core::card_runs_model_kernels(dev)` (the device's capability is the build's) now gates the
      three planners - the prompt split (`split_plan`), the head split (`hs_plan`, `hs_plan_two`)
      and the mixer split (`gs_plan_bw`) - so a cross-capability peer is used only for expert MMQ.
      Rerun the serve start in a serve-stopped window: the three splits must decline, the peer tier
      (experts rows, prompt-chunk rows) must stay, then measure decode and prefill.  The reruns
      declined the splits and the peer tier loaded clean, but the first prompt of a peer serve dies
      in the MMQ with a bare "out of memory": the main card ends loading at 205 MiB free (one-card
      serving runs fine at 211) and the prompt's first MMQ launch takes its module load, scratch
      pool and buffers after the cache is sized; the peer card's price (`cache_size.cpp`) reserves
      its transport buffers but nothing for the peer kernels that load lazily on its first rows.
      Next: MMQ allocation failures now name the ask, the card and its free memory; one run with
      both reserves raised says which card's price is short - price that item where the tier sizes
      itself, and the reserve stays an honest floor.

## Later Ada work (out of the scope above)

- [ ] NVFP4 experts on Ada: decode through the `narrow_bf16` path (done), prefill through an Ada
      design that replaces the `kind::mxf4nvf4` paths - FP8 tensor cores (Ada's full-rate path;
      a numerics move, the quality gate decides) or BF16 to FP32 - not an FP4 emulation; each
      measured before it replaces anything.
- [ ] Ada main card for the other models: inkling2's `attn.cu` calls `this_cluster()` (its
      `attn_chain`-style redesign), hy4flash's tests need `dense_ops`/`mhc` cubins; their other
      files compile clean for sm_89 already.
