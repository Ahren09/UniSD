<div align="center">

# 🧬 UniSD

### *A Unified Self-Distillation Framework for Large Language Models*

[![Website](https://img.shields.io/badge/🌐_Project-Website-2EA44F.svg)](https://unifiedsd.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-2605.06597-b31b1b.svg)](https://arxiv.org/abs/2605.06597)
[![HF Paper](https://img.shields.io/badge/🤗_Hugging_Face-Paper-FFD21E.svg)](https://huggingface.co/papers/2605.06597)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗_Transformers-4.57-FFD21E.svg)](https://huggingface.co/docs/transformers)
[![vLLM](https://img.shields.io/badge/vLLM-0.12-30A14E.svg)](https://github.com/vllm-project/vllm)

[Yiqiao Jin](https://ahren09.github.io)¹\*, [Yiyang Wang](https://hello-diana.github.io/)¹\*, [Lucheng Fu](https://luchengfu6.github.io/)¹, [Yijia Xiao](https://yijia-xiao.com/)², [Yinyi Luo](https://www.linkedin.com/in/yinyi-luo-5b0805324/)³,
[Haoxin Liu](https://scholar.google.com/citations?user=8xaTRNsAAAAJ)¹, [B. Aditya Prakash](https://faculty.cc.gatech.edu/~badityap/)¹, [Josiah Hester](https://josiahhester.com/)¹, [Jindong Wang](https://jd92.wang/)⁴†, [Srijan Kumar](https://faculty.cc.gatech.edu/~srijan/)¹†

¹ Georgia Institute of Technology · ² UCLA · ³ Carnegie Mellon University · ⁴ William & Mary

<sub>\* Equal contribution &nbsp;·&nbsp; † Corresponding authors</sub>

</div>

---

## 📖 Abstract

Self-distillation (SD) offers a promising path for adapting large language models (LLMs) without relying on stronger external teachers. However, SD in autoregressive LLMs remains challenging because self-generated trajectories are free-form, correctness is task-dependent, and plausible rationales can still provide unstable or unreliable supervision. Existing methods mainly examine isolated design choices, leaving their effectiveness, roles, and interactions unclear. In this paper, we propose **UniSD**, a **Uni**fied framework to systematically study **S**elf-**D**istillation. UniSD integrates complementary mechanisms that address supervision reliability, representation alignment, and training stability, including multi-teacher agreement, EMA teacher stabilization, token-level contrastive learning, feature matching, and divergence clipping. Across six benchmarks and six models from three model families, UniSD reveals when self-distillation improves over static imitation, which components drive the gains, and how these components interact across tasks. Guided by these insights, we construct **UniSD\***, an integrated pipeline that combines complementary components and achieves the strongest overall performance, improving over the base model by +5.4 and the strongest baseline by +2.8. Extensive evaluation highlights self-distillation as a practical and steerable approach for efficient LLM adaptation without stronger external teachers.

## ✨ Highlights

- 🧩 **Unified framework** spanning the three axes of self-distillation: supervision reliability, representation alignment, and training stability.
- 🔬 **Five complementary mechanisms** studied in isolation *and* in combination across **6 benchmarks × 6 models × 3 model families**.
- 🏆 **UniSD\*** — the integrated recipe — achieves the strongest overall performance using **only self-derived supervision**, no stronger external teacher required.

## 🧩 The UniSD Framework

UniSD is built from five complementary mechanisms that can be enabled independently or composed into the integrated **UniSD\*** recipe.

| Component | `--mode` | Key flag(s) |
| :--- | :--- | :--- |
| 🤝 **Multi-Teacher Agreement** *(sequence-level)* | `agreement_seq_{random,retrieval,induction}` | `--num-auxiliary-contexts`, `--gamma_agreement` |
| 🎯 **Multi-Teacher Agreement** *(token-level)* | `agreement_tok_{random,retrieval,induction}` | `--num-auxiliary-contexts`, `--gamma_agreement`, `--agreement_stat` |
| 🌊 **EMA Teacher Stabilization** | `ema` | `--ref_model_sync_steps`, `--ref_model_mixup_beta` |
| ⚖️ **Token-Level Contrastive Learning** | `contrastive` | `--contrastive_weight`, `--contrastive_margin` |
| 🧠 **Feature Matching** | `match_joint` / `match_repr` | `--final_layer_distill_weight` |
| ✂️ **Divergence Clipping** *(JSD-Clip)* | `clip` | `--alpha`, `--token_clip` |
| ⭐ **UniSD\*** *(integrated)* | `unisd_star` | combines EMA + matching + contrastive + agreement |

## 🚀 Installation

UniSD targets **Python 3.12 + CUDA 12.8** (cu128 wheels). The install has a few prerequisite steps before the final `pip install -r requirements.txt`, because (a) PyTorch's cu128 build lives on the PyTorch wheel index and (b) flash-attention-2 must be compiled against the installed torch.

```bash
# 1) Create and activate the env
conda create -n unisd python=3.12 -y
conda activate unisd
pip install -U pip setuptools wheel packaging ninja

# 2) Install cu128 PyTorch from the PyTorch wheel index (must precede flash-attn build)
pip install --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0

# 3) Point flash-attn's CUDA build at a 12.x toolkit
#    (on many hosts /usr/local/cuda → 13.x, which mismatches torch's cu128 ABI)
export CUDA_HOME=/usr/local/cuda-12.6

# 4) Install everything else — flash-attn builds from source here (~20 min the first time)
pip install -r requirements.txt --no-build-isolation
```

> 💡 **Don't have `/usr/local/cuda-12.6`?** Any CUDA 12.x toolkit (12.4–12.8) works. Run `ls -d /usr/local/cuda-12*` to see what's available and set `CUDA_HOME` to that path.

> ⚠️ **trl ↔ vLLM compatibility**: this environment ships `trl==1.4.0` (officially supports vLLM 0.12.0–0.18.0) with `vllm==0.20.2`. The combination works in our smoke tests but trl will print a warning at import time. If you hit a runtime error from `VLLMClient`, pin `vllm<0.19`.

### Verify the install

```bash
python -c "
import torch, vllm, flash_attn, flashinfer
print('torch       ', torch.__version__, 'cuda_ok:', torch.cuda.is_available())
print('vllm        ', vllm.__version__)
print('flash_attn  ', flash_attn.__version__)
print('flashinfer  ', flashinfer.__version__)
"
```

Optional environment variables: `WANDB_API_KEY` (logging), `HF_TOKEN` (gated models).

## ⚡ Quick Start

UniSD provides **two ways** to launch training: a high-level orchestrator with sane defaults, and a direct command for full per-flag control.

### Option 1 — Preset orchestrator *(preferred)*

`scripts/run_experiments.py` handles GPU scheduling, dependency-aware sweeps, and sensible defaults.

```bash
# Template
python scripts/run_experiments.py <SUBCOMMAND> [--gpus <GPU_IDS>] [subcommand-flags...]

# Example: token-level contrastive learning
python scripts/run_experiments.py contrastive --weight 0.1 --margin 0.5
```

> 💡 Run `python scripts/run_experiments.py --dry-run` to preview every job before launch.

### Option 2 — Direct command

`python -m src.train.train_unisd` exposes every UniSD flag for fine-grained control.

```bash
# Template
python -m src.train.train_unisd \
    --mode <MODE> --dataset <DATASET> \
    --model_name <MODEL> \
    --per_device_train_batch_size <BATCH> \
    --num-auxiliary-contexts <N> \
    --use_vllm

# Example: token-level contrastive on MBPP with Qwen2.5-7B
python -m src.train.train_unisd \
    --mode contrastive --dataset mbpp \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --per_device_train_batch_size 4 \
    --contrastive_weight 0.1 --use_vllm
```

### Valid placeholder values

| Placeholder | Values |
| :--- | :--- |
| `<SUBCOMMAND>` | `agreement`, `ema`, `contrastive`, `match_joint`, `match_repr`, `clip`, `unisd_star` *(= UniSD\*)*, `induction` |
| `<MODE>` | `agreement_{seq,tok}_{random,retrieval,induction}`, `ema`, `contrastive`, `match_joint`, `match_repr`, `clip`, `unisd_star` |
| `<DATASET>` | `mbpp`, `tooluse`, `scienceqa`, `cos_e`, `medmcqa` *(eval-only: `gpqa`, `humaneval`)* |
| `<MODEL>` | Qwen2.5 (0.5B/1.5B/3B/7B-Instruct), Llama-3.1-8B-Instruct, Gemma-3-4B-IT, InternLM3-8B-Instruct |

### One-time cache prep

A few modes require a one-time cache build:

- 🔻 **`contrastive` and `unisd_star`** need a negative-demonstration cache:
  ```bash
  python -m src.teacher.negative_demonstrations \
      --model_name Qwen/Qwen2.5-7B-Instruct --dataset mbpp
  ```
- 🪄 **`agreement_*_induction`** modes need an induction cache:
  ```bash
  python scripts/run_experiments.py induction --num-demos 5
  ```
- ✅ **`random` and `retrieval`** agreement modes need no prep — embeddings auto-build on first run.

## 📊 Datasets

UniSD is evaluated across **six benchmarks** spanning four task families.

| Dataset | Role | Task |
| :--- | :--- | :--- |
| 🔬 **ScienceQA** | train + eval | Scientific reasoning |
| 💻 **MBPP** | train + eval | Code generation |
| 💭 **CoS-E** | train + eval | Commonsense reasoning |
| 🛠️ **ToolAlpaca** | train + eval | Tool usage |
| 🎓 **GPQA** | OOD eval | Scientific reasoning |
| 🧪 **HumanEval** | OOD eval | Code generation |

## 🤖 Supported Models

UniSD is validated across three model families:

- **Qwen2.5** — 0.5B / 1.5B / 3B / 7B-Instruct *(default: `Qwen/Qwen2.5-7B-Instruct`)*
- **Llama-3.1** — 8B-Instruct
- **Gemma-3** — 4B-IT
- **InternLM3** — 8B-Instruct

## 🧪 Evaluation

Evaluation entry points live under `src/eval/`:

```bash
# Code generation (MBPP / HumanEval)
python -m src.eval.eval_code   --mode <MODE> --dataset humaneval \
    --model_name_or_path <CKPT_OR_HF_ID>

# Multiple-choice QA (ScienceQA / GPQA / CoS-E / MedMCQA)
python -m src.eval.eval_mcqa   --mode <MODE> --dataset gpqa \
    --model_name_or_path <CKPT_OR_HF_ID>

# Tool usage (ToolAlpaca)
python -m src.eval.eval_tooluse --mode <MODE> --dataset tooluse \
    --model_name_or_path <CKPT_OR_HF_ID>
```

## 📝 Citation

If you find UniSD useful in your research, please cite:

```bibtex
@article{jin2026unisd,
  title={UniSD: Towards a Unified Self-Distillation Framework for Large Language Models},
  author={Jin, Yiqiao and Wang, Yiyang and Fu, Lucheng and Xiao, Yijia and Luo, Yinyi and Liu, Haoxin and Prakash, B Aditya and Hester, Josiah and Wang, Jindong and Kumar, Srijan},
  journal={arXiv preprint arXiv:2605.06597},
  year={2026}
}
```

## 🙏 Acknowledgements

UniSD is built on top of excellent open-source work from the community:
[🤗 Transformers](https://github.com/huggingface/transformers) ·
[🤗 TRL](https://github.com/huggingface/trl) ·
[vLLM](https://github.com/vllm-project/vllm) ·
[DeepSpeed](https://github.com/microsoft/DeepSpeed) ·
[PEFT](https://github.com/huggingface/peft) ·
[Accelerate](https://github.com/huggingface/accelerate).

## ⚖️ License

This project is released under the [Apache License 2.0](LICENSE).
