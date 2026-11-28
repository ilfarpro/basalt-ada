<p align="center"><picture><source media="(prefers-color-scheme: dark)" srcset="assets/basalt-lockup-dark.png"><img src="assets/basalt-lockup-light.png" alt="Basalt" width="480"></picture></p>

> **Basalt Ada is a fork of [Basalt](https://github.com/jesdga95/basalt) by [jesdga95](https://github.com/jesdga95). This fork extends hardware support backwards: from Blackwell to Ada Lovelace. Fork is new, only Qwen 3.8 Flash Next UD-Q4_K_XL is implemented and tested. Treat it like project in early Alpha stage. You'll probably encounter errors and problems. Please consider opening issues and PRs to contribute. They are much appreciated!**

*Original Basalt is a local inference engine for large language models: **Qwen3.8-Flash-Next**, **DeepSeek-V4.1-Flash**, **Qwen3.8-27B**, **Hy4-flash** and **Inkling 2 Mini**. It runs
only on an NVIDIA Blackwell (sm_120) GPU on Linux and Windows x64, and on the DGX Spark (GB10, Arm64), with what does not fit in VRAM
(the experts of a mixture-of-experts model) in system RAM, and optionally on more GPUs. The server speaks the OpenAI and Anthropic APIs.*

<p align="center"><img src="docs/img/frontend.gif" alt="The Basalt web UI while a reply streams" width="720"></p>

## Performance

**TL;DR: Basalt Ada offers significantly better prefill speed and exceptional decode performance.**

Tested Qwen3.8 Flash Next UD-Q4_K_XL on Ubuntu 26.04.1 LTS PC with Intel Core i7 14700KF (AVX2), 96GB DDR5 @ 5600 MT/s, RTX 4080 Super PCIe 4.0 x16 and RTX 5060 Ti 16GB PCIe 4.0 x4 with 615.71.09 driver and CUDA 13.4.

> *Goal was to see real world baseline performance numbers without spending few days learning and testing configuration options and their effect on performance. Basalt Ada - settings are from Quick Start section of this README. Strata settings - after thorough tunning option offered in TUI interactive setup process. Tuning took more than hour and returned that default settings were already fastest on this PC (nothing changed).*

### Single GPU

Decode numbers are on prose and prefill is agentic coding session with ~91k prompt that consists of lot's of reasoning traces and some code. Tested on Strata 0.2.23 build 0.1.70 and Basalt Ada 0.5.7 at November 28th, 2026.

| Metric | Basalt Ada | Strata | Result |
| ------- | ----------- | ----- | ------ |
| Prefill | ~4000 t/s | ~950 t/s | Basalt Ada has ~4.2× higher throughput |
| Decode | ~64.5 t/s | ~61.5 t/s | Basalt Ada is ~5.2% faster |

How long it takes to prefill ~91k prompt?

| Engine    | Speed  | Wait time            |
| ---------- | --------- | ---------------- |
| Basalt Ada | 4 000 t/s | **22,8 sec**     |
| Strata     | 950 t/s   | **1 min 36 sec** |

### Dual GPU

| Metric | Basalt Ada | Strata | Result |
| ------- | ----------- | ----- | ------ |
| Prefill | ~4000 t/s | ~2400 t/s | Strata accelerated slowly and reached max ~2400 t/s by the end of the prefill, Basalt Ada delivered ~4000 t/s prefill instantly |
| Decode | ~90.3 t/s | ~43.0 t/s | Basalt Ada is ~2.1× faster |


## Requirements

- Linux x86-64 with AVX2, or Windows 10 1903+ x64, with an NVIDIA Blackwell (compute capability 12.0) or Ada Lovelace (8.9) GPU ([platforms](docs/platforms.md)).
- Or a DGX Spark or another GB10 machine: Linux Arm64 with its integrated Blackwell GPU (12.1) ([Arm64](docs/arm64.md)).
- CUDA 13.
- CMake 3.24+, GCC 13+ (Linux), Python 3.10+, and enough system RAM for the experts that do not fit in VRAM.

## Quick start

Prebuilt Linux (x64 and Arm64) and Windows archives are attached to each [release](https://github.com/jesdga95/basalt/releases) ([what they need](docs/platforms.md#release-archives)); otherwise build from source:

### 1. Clone repo
```sh
git clone https://github.com/ilfarpro/basalt-ada && cd basalt
```
### 2. Choose your build type:
- Ada (sm_89) main card:
  ```sh
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
  ```

- Ada main card with a Blackwell card beside it as the expert tier
  ```sh
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 -DPEER_BLACKWELL=ON
  ```

- Blackwell main card with an Ada (sm_89) card beside it as the expert tier
  ```sh
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPEER_ADA=ON
  ```

### 3. Build
```sh
cmake --build build -j
```
- Executables will be in: bin/basalt, bin/basalt-bench, bin/basalt-probe, bin/basalt-vision
- For Windows: add -G Ninja to the configure line, from an x64 VS 2022 prompt ([platforms](docs/platforms.md#building-on-windows))
- One model only: 
  `-DBASALT_MODELS=qwen4exp, deepseek41, qwen35, hy4flash or inkling2` ([building](docs/building.md#building-only-some-models))


### 4. Install web-server dependencies

You might need to create venv first, if your linux distribution doesn't allow to install packages with pip. 
```sh
python3 -m venv .venv

source .venv/bin/activate
```
Then install dependecies
```sh
pip install -r requirements.txt
```
Activate venv
```sh
source .venv/bin/activate
```

### 5. Download a model
You also can [repack existing GGUF file](docs/model-file.md).
```sh
hf download jesdga/Qwen3.8-Flash-Next-Basalt Qwen3.8-FN-UD-Q4_K_XL.basalt --local-dir models
```

### 6. Configure

Put the settings in `basalt_config.json`:

```json
{
  "version": 2,
  "models": [
    {
      "model": "models/Qwen3.8-FN-UD-Q4_K_XL.basalt",
      "gpu": "0,1",
      "gpu-mode": "peer",
      "kv": "int8"
    }
  ],
  "engine": {
    "resident-experts": true
  },
  "server": {
    "port": 8080
  }
}
```
Every setting: [configuration](docs/configuration/index.md). More examples:
[example configurations](docs/configuration/examples.md).

### 7. Launch
```sh
bin/basalt serve --config basalt_config.json
```

The server listens on 127.0.0.1 and 8080 by default. Override with `--host 0.0.0.0` to open it from local network and change port with `--port` if neccessary. Provide optional `--api-key` when opening access from local network.

```sh
bin/basalt serve --config basalt_config.json --host 0.0.0.0 --port 8080 --api-key <your-api-key-here>
```

## Documentation

- **Run:** [running](docs/running.md), [configuration](docs/configuration/index.md), [models and the router](docs/configuration/models.md),
  [sampling](docs/configuration/sampling.md), [image sources](docs/configuration/image-sources.md),
  [documents](docs/configuration/documents.md),
  [diagnostics](docs/configuration/diagnostics.md), [basalt-probe](docs/probe.md), [platforms](docs/platforms.md).
- **Models:** [models](docs/models/index.md), [Flash-Next](docs/models/qwen38-flash-next.md), [DeepSeek](docs/models/deepseek41.md),
  [Qwen3.8-27B](docs/models/qwen38-27b.md), [Hy4-flash](docs/models/hy4-flash.md), [Inkling 2 Mini](docs/models/inkling2.md),
  [the `.basalt` file](docs/model-file.md), [file format changes](docs/basalt-file-changelog.md).
- **Quality:** [precision](docs/precision/index.md), [method](docs/precision/quality.md).
- **Build and develop:** [building](docs/building.md), [adding a model](docs/adding-a-model.md), [community runs](docs/community.md),
  [changelog](docs/changelog/index.md), [contributing](CONTRIBUTING.md).

## Credits

- **Code:** [Basalt](https://github.com/jesdga95/basalt) (MIT) the project Basalt Ada forks; [Strata](https://github.com/Niko1221/Strata), the project original Basalt forks (MIT); [Infernix](https://github.com/Wallawalla47/Infernix)
  (Ian Ranson and contributors, Apache-2.0), whose kernels are adapted; [ggml and llama.cpp](https://github.com/ggml-org/llama.cpp)
  (MIT); and ideas from [NInfer](https://github.com/Neroued/ninfer). Licences: [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES).
- **Models:** [Qwen](https://huggingface.co/Qwen), [DeepSeek](https://huggingface.co/deepseek-ai), [Tencent](https://huggingface.co/tencent) (Hy4-flash) and Thinking Machines Lab (Inkling 2 Mini), under the licences on their cards;
  [Unsloth](https://huggingface.co/unsloth)'s GGUFs; [NVIDIA](https://huggingface.co/nvidia)'s NVFP4 checkpoints;
  [QUASAR-QAT](https://huggingface.co/QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4)'s NVFP4 27B (Apache-2.0); [z-lab](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)'s DFlash 2 drafter.

Thank you to everyone who has [contributed](https://github.com/jesdga95/basalt/graphs/contributors) to the project.

## License

MIT, see [LICENSE](LICENSE).
