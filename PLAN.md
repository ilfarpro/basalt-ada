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
- [ ] `narrow_bf16`: an sm_89 body for the E2M1/E4M3 to bf16x2 path (`cvt.*.e2m1x2` and
      `mul.rn.bf16x2` are Blackwell/sm_90+; keep the conversion exact); `narrow_bf16_parity`
      runs on sm_89.
- [ ] Decode: `window_gemm` on sm_89 - the bf16 MMA with FP32 accumulation ports as is; the TMA
      loads get a `cp.async` body designed for Ada's load pipeline (stage count, tile shape and
      SM occupancy tuned on the card, not inherited from sm_120).
- [ ] TMA audit: a `cp.async` body wherever the main path uses `cp.async.bulk` (`dense_tc`,
      `verify_kernels`, `qsa_mid`, `dec_moe`, `attn_chain`, `hc_fused`, the prefill GEMMs,
      inkling2/hy4flash ops); thread-block clusters only through the SPLIT instantiation pattern,
      re-tuned for Ada's SM count.
- [ ] Prefill: replace the `kind::mxf4nvf4` paths with an Ada design, not an FP4 emulation: an
      FP8 tensor-core GEMM (Ada's full-rate path; a numerics move, the quality gate decides) or
      BF16 to FP32.
- [ ] First milestone: a UD-Q4_K_XL Flash-Next pack runs end to end on a 4080-class card.
- [ ] NVFP4 experts on Ada come last: decode through the `narrow_bf16` path, prefill through the
      Ada prefill path, each measured before it replaces anything.
- [ ] Benchmarks: Basalt vs Strata on one card, one pack, one drafter; ABBA with the power limit
      stated; decode in ms per window at T=1/4/8/16 with the acceptance rate; stock drivers only.
