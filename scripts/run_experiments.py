#!/usr/bin/env python3
"""Unified experiment runner with dependency-aware GPU scheduler.

All jobs from all experiments are materialized into a single dependency graph.
A continuous event loop dispatches the next runnable job whenever a GPU becomes free.

Usage:
    python scripts/run_experiments.py                     # run UNISD_STAR experiments
    python scripts/run_experiments.py --dry-run            # preview all jobs
    python scripts/run_experiments.py induction --gamma 0.01 --nctx 3 --num-demos 3
    python scripts/run_experiments.py agreement --modes agreement_seq_random,agreement_seq_induction
    python scripts/run_experiments.py agreement --modes agreement_tok_random,agreement_tok_retrieval,agreement_tok_induction --gpus 0,1
    python scripts/run_experiments.py ema --sync-steps 10 --mixup-beta 0.9
    python scripts/run_experiments.py contrastive --weight 0.1 --margin 0.5
    python scripts/run_experiments.py unisd_star          # combined recipe
"""

import argparse
import glob
import os
import re
import sys
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.const import *  # noqa: E402
from src.utils.path_utils import get_model_dir, get_induction_cache_path, get_neg_demo_cache_path  # noqa: E402

from gpu_pool import GPUPool, Job, JobNode, JobGraph, DependencyScheduler  # noqa: E402

# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_MODELS = [
    # "Qwen/Qwen2.5-0.5B-Instruct",
    # "Qwen/Qwen2.5-1.5B-Instruct",
    # "Qwen/Qwen2.5-3B-Instruct",
    # "Qwen/Qwen3-8B",
    # "google/gemma-3-4b-it",
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    # "internlm/internlm3-8b-instruct",
    # "Qwen/Qwen3.5-9B",
]

# Default datasets per experiment
# DEFAULT_DATASETS = ["mbpp", "scienceqa-all-max1000", "cos_e-max1000", "tooluse"]
# DEFAULT_DATASETS = ["mbpp", "tooluse"] #, "scienceqa-all", "cos_e"]
DEFAULT_DATASETS = [MBPP, TOOLUSE, "scienceqa-all", COS_E, MEDMCQA]

ALL_EXPERIMENTS = [AGREEMENT, EMA, CONTRASTIVE, MATCH_REPR, UNISD_STAR]

# Set in main() from --gpu-memory-utilization. None = keep existing defaults.
_GPU_MEM_UTIL: float | None = None
# Set in main() from --no-vllm-sleep. If True, append --no-vllm_enable_sleep_mode
# to every training command (workaround for drivers whose cuMem sleep is broken).
_DISABLE_VLLM_SLEEP: bool = False


def _vllm_gmu_args() -> list[str]:
    """Append global vLLM flags injected on every training command.

    - --vllm_gpu_memory_utilization X (iff overridden via CLI)
    - --no-vllm_enable_sleep_mode    (iff --no-vllm-sleep is passed; works around
      'CUDA driver version is insufficient for CUDA runtime version' errors from
      vllm/device_allocator/cumem.py on drivers without cuMem sleep support).
    """
    extras: list[str] = []
    if _GPU_MEM_UTIL is not None:
        extras += ["--vllm_gpu_memory_utilization", str(_GPU_MEM_UTIL)]
    if _DISABLE_VLLM_SLEEP:
        extras += ["--no-vllm_enable_sleep_mode"]
    return extras

# TEACHER_AGREEMENT_MODES = [AGREEMENT_TOK_RANDOM, AGREEMENT_TOK_RETRIEVAL, AGREEMENT_TOK_INDUCTION, AGREEMENT_SEQ_RANDOM, AGREEMENT_SEQ_RETRIEVAL, AGREEMENT_SEQ_INDUCTION]

# TODO: Add more modes here
TEACHER_AGREEMENT_MODES = [AGREEMENT_TOK_RETRIEVAL, AGREEMENT_SEQ_RETRIEVAL]


# Batch sizes keyed by model short name -> method -> batch_size
BATCH_SIZES = {
    "Qwen2.5-0.5B-Instruct": {"default": 16},
    "Qwen2.5-1.5B-Instruct": {"default": 8},
    "Qwen2.5-3B-Instruct":   {"default": 8},
    "Qwen2.5-7B-Instruct":   {"default": 4},
    "Qwen3-8B":              {"default": 4},
    "gemma-3-4b-it":         {"default": 8},
    "Llama-3.1-8B-Instruct": {"default": 4},
    "internlm3-8b-instruct": {"default": 4},
    "Qwen3.5-9B":            {"default": 4},
}

# Datasets that are eval-only and must never be used for training
EVAL_ONLY_DATASETS = {"gpqa_main", "humaneval"}

# Cross-eval mapping: source_base_dataset -> (target_dataset, log_name)
CROSS_EVAL_MAP = {
    "scienceqa-all": ("gpqa_main", "eval_gpqa.log"),
    "mbpp": ("humaneval", "eval_humaneval.log"),
}

# ── Shared Helpers ───────────────────────────────────────────────────────────


def filter_trainable_datasets(datasets: list[str], experiment: str) -> list[str]:
    """Remove eval-only datasets (GPQA, HumanEval) from training dataset list."""
    trainable = []
    for ds in datasets:
        base, _ = parse_dataset_max(ds)
        if base in EVAL_ONLY_DATASETS:
            print(f"WARNING: skipping eval-only dataset '{ds}' for {experiment} training")
        else:
            trainable.append(ds)
    return trainable


def model_short(model: str) -> str:
    return os.path.basename(model)


def append_resume_args(cmd: list[str], max_steps: int = -1,
                       resume_from_checkpoint: str | None = None,
                       additional_epochs: float = 0) -> None:
    """Append checkpoint resume and training length args to a command."""
    if resume_from_checkpoint:
        cmd += ["--resume_from_checkpoint", resume_from_checkpoint]
    if additional_epochs > 0:
        cmd += ["--additional_epochs", str(additional_epochs)]
    elif max_steps > 0:
        cmd += ["--max_steps", str(max_steps)]


def parse_dataset_max(dataset: str) -> tuple[str, int | None]:
    """Split 'cos_e-max1000' -> ('cos_e', 1000), 'scienceqa-all' -> ('scienceqa-all', None)."""
    m = re.match(r'^(.+)-max(\d+)$', dataset)
    if m:
        return m.group(1), int(m.group(2))
    return dataset, None


def get_train_dataset(dataset: str) -> str:
    """Map display dataset name to training dataset name."""
    return {"gpqa_main": "gpqa", "scienceqa-all": "scienceqa"}.get(dataset, dataset)


# Maps display dataset base names to (dataset, subset) for get_model_dir
DATASET_SUBSETS = {
    "scienceqa-all": ("scienceqa", "all"),
    "gpqa_main": ("gpqa", "gpqa_main"),
}


def parse_display_dataset(display_name: str) -> tuple[str, str | None, int | None]:
    """Parse 'scienceqa-all-max1000' -> ('scienceqa', 'all', 1000)."""
    base, max_samples = parse_dataset_max(display_name)
    dataset, subset = DATASET_SUBSETS.get(base, (base, None))
    return dataset, subset, max_samples


def get_dataset_args(train_dataset: str, experiment: str) -> list[str]:
    """Dataset-specific training args."""
    if train_dataset == "mbpp":
        return ["--mbpp_config", "full"]
    if train_dataset == "tooluse":
        return ["--mask_truncated_completions"]
    return []


def get_hparam_tag(mode: str, **kwargs) -> str:
    """Build hparam subdirectory tag. Mirrors src/utils/path_utils.build_hparam_tag."""
    if mode == MATCH_JOINT:
        return f"weight{kwargs['final_layer_distill_weight']}"
    if mode == CONTRASTIVE:
        return f"weight{kwargs['contrastive_weight']}_margin{kwargs['contrastive_margin']}"
    if mode == EMA:
        return f"sync{kwargs['ref_model_sync_steps']}_beta{kwargs['ref_model_mixup_beta']}"
    if mode == CLIP:
        clip_q = kwargs.get('token_clip_quantile', None)
        if clip_q is not None:
            return f"clipq{clip_q}_alpha{kwargs.get('alpha', 0.0)}"
        clip = kwargs.get('token_clip', 10.0)
        return f"clip{clip}_alpha{kwargs.get('alpha', 0.0)}"
    if mode == UNISD_STAR:
        fm = kwargs.get('fewshot_method', RANDOM)
        return (f"{fm}_gamma{kwargs['gamma_agreement']}_nctx{kwargs['num_auxiliary_contexts']}"
                f"_cw{kwargs['contrastive_weight']}_margin{kwargs['contrastive_margin']}"
                f"_sync{kwargs['ref_model_sync_steps']}_ema{kwargs['ref_model_mixup_beta']}"
                f"_fl{kwargs['final_layer_distill_weight']}")
    # fewshot / induction modes
    return f"gamma{kwargs['gamma_agreement']}_nctx{kwargs['num_auxiliary_contexts']}"


def get_out_dir(mode: str, dataset: str, model: str, hparam_tag: str = "") -> str:
    """Build output directory path (delegates to data_utils.get_model_dir).

    New training writes under outputs/unisd/<mode>/... so existing pre-rename
    checkpoints under outputs/<old_mode>/ are not disturbed.
    """
    ds, subset, max_samples = parse_display_dataset(dataset)
    return str(get_model_dir("outputs/unisd", mode, ds, model, subset=subset,
                             hparam_tag=hparam_tag, max_samples=max_samples))


def has_results(directory: str) -> bool:
    return bool(glob.glob(os.path.join(directory, "results*.xlsx")))


def find_latest_checkpoint(directory: str) -> str | None:
    """Find latest checkpoint-* directory, version-sorted."""
    ckpts = []
    for p in glob.glob(os.path.join(directory, "checkpoint-*")):
        try:
            int(os.path.basename(p).split("-")[1])
            ckpts.append(p)
        except ValueError:
            continue
    if not ckpts:
        return None
    ckpts.sort(key=lambda p: int(os.path.basename(p).split("-")[1]))
    return ckpts[-1]


def get_batch_size(model: str, experiment: str, dataset: str = "") -> int:
    """Get batch size for a model based on experiment type."""
    ms = model_short(model)
    sizes = BATCH_SIZES.get(ms, {})
    bsz = sizes.get(experiment, sizes.get("default", 4))
    return bsz


# ── Eval Command Builder ────────────────────────────────────────────────────


def build_eval_cmd(dataset: str, mode: str, ckpt: str, hparam_tag: str = "",
                   ms: str = "") -> list[str]:
    """Build evaluation command for a dataset.

    Args:
        dataset: Display dataset name (mbpp, scienceqa-all, gpqa_main, tooluse, cos_e),

        mode: Training mode
        ckpt: Checkpoint path
        hparam_tag: Hparam subdirectory tag
        ms: model_short (needed for tooluse with no hparam_tag)
    """
    base_dataset, max_samples = parse_dataset_max(dataset)

    common = [
        "--mode", mode,
        "--model_name_or_path", ckpt,
        "--output_dir", "outputs/unisd",
        "--temperature", "0.0",
        "--max_new_tokens", "1024",
        "--gpu_memory_utilization", str(_GPU_MEM_UTIL if _GPU_MEM_UTIL is not None else 0.6),
    ]
    if hparam_tag:
        common += ["--hparam-tag", hparam_tag]
    if max_samples is not None:
        common += ["--suffix", f"max{max_samples}"]
        common += ["--max-samples", str(max_samples)]

    if base_dataset == "mbpp":
        return ["python", "src/eval/eval_code.py"] + common + [
            "--dataset", "mbpp", "--split", "test", "--max_prompt_length", "2048",
        ]
    elif base_dataset == "humaneval":
        return ["python", "src/eval/eval_code.py"] + common + [
            "--dataset", "humaneval", "--split", "test", "--max_prompt_length", "2048",
        ]
    elif base_dataset == "scienceqa-all":
        return ["python", "-m", "src.eval.eval_mcqa"] + common + [
            "--dataset", "scienceqa", "--subset", "all",
            "--split", "test", "--max_prompt_length", "4096",
        ]
    elif base_dataset == "gpqa_main":
        return ["python", "-m", "src.eval.eval_mcqa"] + common + [
            "--dataset", "gpqa", "--subset", "gpqa_main",
            "--split", "all", "--max_prompt_length", "4096",
        ]
    elif base_dataset == "cos_e":
        return ["python", "-m", "src.eval.eval_mcqa"] + common + [
            "--dataset", "cos_e", "--split", "validation",
            "--max_prompt_length", "4096",
        ]
    elif base_dataset == "medmcqa":
        return ["python", "-m", "src.eval.eval_mcqa"] + common + [
            "--dataset", "medmcqa", "--split", "validation",
            "--max_prompt_length", "4096",
        ]
    elif base_dataset == "tooluse":
        return ["python", "-m", "src.eval.eval_tooluse"] + common + [
            "--dataset", "tooluse", "--split", "test",
        ]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


# ── Deferred Eval Command ──────────────────────────────────────────────────


def _resolve_eval_cmd(dataset, mode, out_dir, hparam_tag, ms):
    """Deferred eval command builder -- called when checkpoint becomes available."""
    ckpt = find_latest_checkpoint(out_dir)
    if not ckpt:
        raise FileNotFoundError(f"No checkpoint in {out_dir}")
    return build_eval_cmd(dataset, mode, ckpt, hparam_tag, ms)


# ── Eval Node Builder (shared across experiments) ─────────────────────────


def _build_eval_nodes(dataset: str, mode: str, models: list[str],
                      hparam_tag: str,
                      train_nodes_by_model: dict[str, str],
                      experiment: str = "") -> list[JobNode]:
    """Build eval + cross-eval JobNodes for one dataset across all models.

    Args:
        train_nodes_by_model: dict mapping model_short -> train JobNode name,
                              used for depends_on wiring.
    """
    nodes = []
    for model in models:
        ms = model_short(model)
        out_dir = get_out_dir(mode, dataset, model, hparam_tag)
        train_name = train_nodes_by_model[ms]
        eval_name = f"eval_{mode}_{dataset}_{ms}"
        meta = {"mode": mode, "dataset": dataset, "model": model,
                "out_dir": out_dir, "hparam_tag": hparam_tag}

        if has_results(out_dir):
            nodes.append(JobNode(name=eval_name, status="skipped", phase="eval",
                                 experiment=experiment, metadata=meta))
        else:
            ckpt = find_latest_checkpoint(out_dir)
            if not ckpt:
                final_dir = os.path.join(out_dir, "final")
                if os.path.isdir(final_dir):
                    ckpt = final_dir
            if ckpt:
                # Checkpoint exists now -- build cmd immediately
                cmd = build_eval_cmd(dataset, mode, ckpt, hparam_tag, ms)
                nodes.append(JobNode(name=eval_name, cmd=cmd,
                                     depends_on=[train_name], phase="eval",
                                     experiment=experiment, metadata=meta,
                                     log_file=os.path.join(out_dir, "eval.log")))
            else:
                # Defer -- checkpoint will exist after training
                nodes.append(JobNode(name=eval_name,
                                     cmd_factory=partial(_resolve_eval_cmd, dataset, mode, out_dir, hparam_tag, ms),
                                     depends_on=[train_name], phase="eval",
                                     experiment=experiment, metadata=meta,
                                     log_file=os.path.join(out_dir, "eval.log")))

        # Cross-eval
        eval_skipped = has_results(out_dir)
        cross_eval_added = False
        base_ds, _ = parse_dataset_max(dataset)
        if base_ds in CROSS_EVAL_MAP:
            target_dataset, log_name = CROSS_EVAL_MAP[base_ds]
            cross_name = f"eval_{mode}_{dataset}-{target_dataset}_{ms}"
            target_out = get_out_dir(mode, target_dataset, model, hparam_tag)
            src_dir = get_out_dir(mode, dataset, model, hparam_tag)
            cross_meta = {**meta, "source_dataset": dataset, "target_dataset": target_dataset}

            if has_results(target_out):
                nodes.append(JobNode(name=cross_name, status="skipped", phase="eval",
                                     experiment=experiment, metadata=cross_meta))
            else:
                cross_eval_added = True
                # Cross-eval depends on SOURCE train job (same checkpoint)
                nodes.append(JobNode(name=cross_name,
                                     cmd_factory=partial(_resolve_eval_cmd, target_dataset, mode, src_dir, hparam_tag, ms),
                                     depends_on=[train_name], phase="eval",
                                     experiment=experiment, metadata=cross_meta,
                                     log_file=os.path.join(src_dir, log_name)))

        # Cleanup: delete checkpoint and final dirs after all evals finish
        if eval_skipped:
            nodes.append(JobNode(
                name=f"cleanup_{eval_name}", status="skipped",
                phase="cleanup", experiment=experiment))
        else:
            cleanup_deps = [eval_name]
            if cross_eval_added:
                cleanup_deps.append(cross_name)
            cleanup_cmd = [
                "bash", "-c",
                f"rm -rf {out_dir}/checkpoint-* {out_dir}/final 2>/dev/null; true"]
            nodes.append(JobNode(
                name=f"cleanup_{eval_name}", cmd=cleanup_cmd,
                depends_on=cleanup_deps, phase="cleanup",
                experiment=experiment))

    return nodes


# ══════════════════════════════════════════════════════════════════════════════
# Per-Experiment Builders (return list[JobNode] for all phases)
# ══════════════════════════════════════════════════════════════════════════════


def build_induction(models: list[str], datasets: list[str],
                    gamma: float = 0.01, nctx: int = 5, num_demos: int = 3,
                    max_steps: int = -1, **kwargs) -> list[JobNode]:
    """Induction experiment builder -- all phases."""
    datasets = filter_trainable_datasets(datasets, "induction")
    hparam_tag = get_hparam_tag(AGREEMENT_SEQ_INDUCTION,
                                gamma_agreement=gamma,
                                num_auxiliary_contexts=nctx)
    nodes = []
    for dataset in datasets:
        base_dataset, max_samples = parse_dataset_max(dataset)
        train_ds = get_train_dataset(base_dataset)
        train_nodes_by_model = {}

        for model in models:
            ms = model_short(model)
            cache_name = f"cache_induction_{train_ds}_{ms}"
            train_name = f"induction_train_{dataset}_{ms}"
            train_nodes_by_model[ms] = train_name
            out_dir = get_out_dir(AGREEMENT_SEQ_INDUCTION, dataset, model, hparam_tag)
            meta = {"mode": AGREEMENT_SEQ_INDUCTION, "dataset": dataset, "model": model,
                    "out_dir": out_dir, "hparam_tag": hparam_tag}

            # Cache node
            cache_path = get_induction_cache_path("outputs", model, train_ds, nctx, num_demos)
            cache_cmd = [
                "python", "-m", "src.teacher.instruction_induction",
                "--dataset", train_ds,
                "--model_name", model,
                "--num-auxiliary-contexts", str(nctx),
                "--induction_num_demos", str(num_demos),
                "--output_dir", "outputs",
                "--seed", "42",
            ]
            if cache_path.exists():
                nodes.append(JobNode(name=cache_name, cmd=cache_cmd, status="skipped",
                                     phase="cache", experiment="induction",
                                     metadata={"dataset": train_ds, "model": model}))
            else:
                nodes.append(JobNode(name=cache_name, cmd=cache_cmd, phase="cache",
                                     experiment="induction",
                                     log_file=f"logs/induction_cache_{train_ds}_{ms}.log",
                                     metadata={"dataset": train_ds, "model": model}))

            # Train node (depends on cache)
            if find_latest_checkpoint(out_dir):
                nodes.append(JobNode(name=train_name, status="skipped",
                                     depends_on=[cache_name],
                                     phase="train", experiment="induction", metadata=meta))
            else:
                bsz = get_batch_size(model, "induction")
                cmd = [
                    "python", "-m", "src.train.train_unisd",
                    "--mode", AGREEMENT_SEQ_INDUCTION,
                    "--dataset", train_ds,
                    "--model_name", model,
                    "--output_dir", "outputs/unisd",
                    "--max_completion_length", "1024",
                    "--max_prompt_length", "3072",
                    "--per_device_train_batch_size", str(bsz),
                    "--gradient_accumulation_steps", "4",
                    "--num-auxiliary-contexts", str(nctx),
                    "--induction_num_demos", str(num_demos),
                    "--gamma_agreement", str(gamma),
                    "--use_vllm",
                    "--temperature", "0.7",

                ] + _vllm_gmu_args() + get_dataset_args(train_ds, "induction")

                if max_samples is not None:
                    cmd += ["--num_train_examples", str(max_samples)]
                append_resume_args(cmd, max_steps, kwargs.get("resume_from_checkpoint"),
                                   kwargs.get("additional_epochs", 0))

                nodes.append(JobNode(name=train_name, cmd=cmd,
                                     depends_on=[cache_name],
                                     phase="train", experiment="induction", metadata=meta,
                                     log_file=os.path.join(out_dir, "train.log")))

        nodes += _build_eval_nodes(dataset, AGREEMENT_SEQ_INDUCTION, models, hparam_tag,
                                   train_nodes_by_model, experiment="induction")

    return nodes


def build_teacher_agreement(models: list[str], datasets: list[str],
                            modes: list[str] = None,
                            gamma: float = 1.0, nctx: int = 3, num_demos: int = 3,
                            max_steps: int = -1, **kwargs) -> list[JobNode]:
    """Teacher agreement experiment builder (multiple modes) -- all phases."""
    datasets = filter_trainable_datasets(datasets, AGREEMENT)
    if modes is None:
        modes = TEACHER_AGREEMENT_MODES
    hparam_tag = get_hparam_tag(AGREEMENT_SEQ_RANDOM,  # all these modes use same tag format
                                gamma_agreement=gamma,
                                num_auxiliary_contexts=nctx)

    nodes = []
    for dataset in datasets:
        base_dataset, _ = parse_dataset_max(dataset)
        train_ds = get_train_dataset(base_dataset)
        _, max_samples = parse_dataset_max(dataset)

        for model in models:
            ms = model_short(model)

            # Cache node (shared with induction -- deduped by JobGraph.add())
            cache_name = f"cache_induction_{train_ds}_{ms}"
            cache_path = get_induction_cache_path("outputs", model, train_ds, nctx, num_demos)
            cache_cmd = [
                "python", "-m", "src.teacher.instruction_induction",
                "--dataset", train_ds,
                "--model_name", model,
                "--num-auxiliary-contexts", str(nctx),
                "--induction_num_demos", str(num_demos),
                "--output_dir", "outputs",
                "--seed", "42",
            ]
            # Note: induction-cache --output_dir stays as "outputs" so the cache
            # file (outputs/cache/induction_*) is shared across pre/post-rename runs.
            if cache_path.exists():
                nodes.append(JobNode(name=cache_name, cmd=cache_cmd, status="skipped",
                                     phase="cache", experiment=AGREEMENT,
                                     metadata={"dataset": train_ds, "model": model}))
            else:
                nodes.append(JobNode(name=cache_name, cmd=cache_cmd, phase="cache",
                                     experiment=AGREEMENT,
                                     log_file=f"logs/induction_cache_{train_ds}_{ms}.log",
                                     metadata={"dataset": train_ds, "model": model}))

        # Train + eval nodes per mode
        for mode in modes:
            train_nodes_by_model = {}

            for model in models:
                ms = model_short(model)
                cache_name = f"cache_induction_{train_ds}_{ms}"
                out_dir = get_out_dir(mode, dataset, model, hparam_tag)
                train_name = f"{mode}_train_{dataset}_{ms}"
                train_nodes_by_model[ms] = train_name
                meta = {"mode": mode, "dataset": dataset, "model": model,
                        "out_dir": out_dir, "hparam_tag": hparam_tag}

                if find_latest_checkpoint(out_dir):
                    nodes.append(JobNode(name=train_name, status="skipped",
                                         depends_on=[cache_name],
                                         phase="train", experiment=AGREEMENT,
                                         metadata=meta))
                else:
                    bsz = get_batch_size(model, AGREEMENT)
                    mode_args = [
                        "--num-auxiliary-contexts", str(nctx),
                        "--gamma_agreement", str(gamma),
                    ]
                    if mode in (AGREEMENT_SEQ_INDUCTION, AGREEMENT_TOK_INDUCTION):
                        mode_args += ["--induction_num_demos", str(num_demos)]

                    cmd = [
                        "python", "-m", "src.train.train_unisd",
                        "--mode", mode,
                        "--dataset", train_ds,
                        "--model_name", model,
                        "--output_dir", "outputs/unisd",
                        "--max_completion_length", "1024",
                        "--max_prompt_length", "3072",
                        "--per_device_train_batch_size", str(bsz),
                        "--gradient_accumulation_steps", "4",
                        "--use_vllm",
                        "--temperature", "0.7",

                    ] + _vllm_gmu_args() + mode_args + get_dataset_args(train_ds, AGREEMENT)

                    if max_samples is not None:
                        cmd += ["--num_train_examples", str(max_samples)]
                    append_resume_args(cmd, max_steps, kwargs.get("resume_from_checkpoint"),
                                       kwargs.get("additional_epochs", 0))

                    nodes.append(JobNode(name=train_name, cmd=cmd,
                                         depends_on=[cache_name],
                                         phase="train", experiment=AGREEMENT,
                                         metadata=meta,
                                         log_file=os.path.join(out_dir, "train.log")))

            nodes += _build_eval_nodes(dataset, mode, models, hparam_tag,
                                       train_nodes_by_model, experiment=AGREEMENT)

    return nodes


def build_teacher_ema(models: list[str], datasets: list[str],
                      sync_steps: int = 10, mixup_beta: float = 0.9,
                      max_steps: int = -1, **kwargs) -> list[JobNode]:
    """Teacher EMA experiment builder -- all phases."""
    datasets = filter_trainable_datasets(datasets, "teacher_ema")
    hparam_tag = get_hparam_tag(EMA,
                                ref_model_sync_steps=sync_steps,
                                ref_model_mixup_beta=mixup_beta)
    nodes = []
    for dataset in datasets:
        base_dataset, max_samples = parse_dataset_max(dataset)
        train_ds = get_train_dataset(base_dataset)
        train_nodes_by_model = {}

        for model in models:
            ms = model_short(model)
            out_dir = get_out_dir(EMA, dataset, model, hparam_tag)
            train_name = f"teacher_ema_train_{dataset}_{ms}"
            train_nodes_by_model[ms] = train_name
            meta = {"mode": EMA, "dataset": dataset, "model": model,
                    "out_dir": out_dir, "hparam_tag": hparam_tag}

            if find_latest_checkpoint(out_dir):
                nodes.append(JobNode(name=train_name, status="skipped",
                                     phase="train", experiment="teacher_ema", metadata=meta))
            else:
                bsz = get_batch_size(model, EMA)
                cmd = [
                    "python", "-m", "src.train.train_unisd",
                    "--mode", EMA,
                    "--dataset", train_ds,
                    "--model_name", model,
                    "--output_dir", "outputs/unisd",
                    "--max_completion_length", "1024",
                    "--max_prompt_length", "3072",
                    "--per_device_train_batch_size", str(bsz),
                    "--gradient_accumulation_steps", "4",
                    "--ref_model_sync_steps", str(sync_steps),
                    "--ref_model_mixup_beta", str(mixup_beta),
                    "--use_vllm",
                    "--temperature", "0.7",

                ] + _vllm_gmu_args() + get_dataset_args(train_ds, EMA)

                if max_samples is not None:
                    cmd += ["--num_train_examples", str(max_samples)]
                append_resume_args(cmd, max_steps, kwargs.get("resume_from_checkpoint"),
                                   kwargs.get("additional_epochs", 0))

                nodes.append(JobNode(name=train_name, cmd=cmd, phase="train",
                                     experiment="teacher_ema", metadata=meta,
                                     log_file=os.path.join(out_dir, "train.log")))

        nodes += _build_eval_nodes(dataset, EMA, models, hparam_tag,
                                   train_nodes_by_model, experiment="teacher_ema")

    return nodes


def build_token_cl(models: list[str], datasets: list[str],
                   weight: float = 0.1, margin: float = 0.5,
                   max_steps: int = -1, **kwargs) -> list[JobNode]:
    """Token contrastive learning experiment builder -- all phases."""
    datasets = filter_trainable_datasets(datasets, "token_cl")
    hparam_tag = get_hparam_tag(CONTRASTIVE,
                                contrastive_weight=weight,
                                contrastive_margin=margin)
    nodes = []
    for dataset in datasets:
        base_dataset, max_samples = parse_dataset_max(dataset)
        train_ds = get_train_dataset(base_dataset)
        train_nodes_by_model = {}

        for model in models:
            ms = model_short(model)
            out_dir = get_out_dir(CONTRASTIVE, dataset, model, hparam_tag)
            cache_name = f"negdemo_{train_ds}_{ms}"
            train_name = f"token_cl_train_{dataset}_{ms}"
            train_nodes_by_model[ms] = train_name
            meta = {"mode": CONTRASTIVE, "dataset": dataset, "model": model,
                    "out_dir": out_dir, "hparam_tag": hparam_tag}

            # Cache node (neg demos)
            cache_path = get_neg_demo_cache_path("outputs/cache/demonstration", train_ds, model, None)
            if cache_path.exists():
                nodes.append(JobNode(name=cache_name, status="skipped",
                                     phase="cache", experiment="token_cl",
                                     metadata={"dataset": train_ds, "model": model}))
            else:
                cache_cmd = [
                    "python", "-m", "src.teacher.negative_demonstrations",
                    "--model_name", model,
                    "--dataset", train_ds,
                ]
                nodes.append(JobNode(name=cache_name, cmd=cache_cmd, phase="cache",
                                     experiment="token_cl",
                                     log_file=f"logs/negdemo_{train_ds}_{ms}.log",
                                     metadata={"dataset": train_ds, "model": model}))

            # Train node (depends on cache)
            if find_latest_checkpoint(out_dir):
                nodes.append(JobNode(name=train_name, status="skipped",
                                     depends_on=[cache_name],
                                     phase="train", experiment="token_cl", metadata=meta))
            else:
                bsz = get_batch_size(model, CONTRASTIVE)
                cmd = [
                    "python", "-m", "src.train.train_unisd",
                    "--mode", CONTRASTIVE,
                    "--dataset", train_ds,
                    "--model_name", model,
                    "--output_dir", "outputs/unisd",
                    "--max_completion_length", "1024",
                    "--max_prompt_length", "3072",
                    "--per_device_train_batch_size", str(bsz),
                    "--gradient_accumulation_steps", "4",
                    "--contrastive_weight", str(weight),
                    "--contrastive_margin", str(margin),
                    "--use_vllm",
                    "--temperature", "0.7",

                ] + _vllm_gmu_args() + get_dataset_args(train_ds, CONTRASTIVE)

                if max_samples is not None:
                    cmd += ["--num_train_examples", str(max_samples)]
                append_resume_args(cmd, max_steps, kwargs.get("resume_from_checkpoint"),
                                   kwargs.get("additional_epochs", 0))

                nodes.append(JobNode(name=train_name, cmd=cmd,
                                     depends_on=[cache_name],
                                     phase="train", experiment="token_cl", metadata=meta,
                                     log_file=os.path.join(out_dir, "train.log")))

        nodes += _build_eval_nodes(dataset, CONTRASTIVE, models, hparam_tag,
                                   train_nodes_by_model, experiment="token_cl")

    return nodes


def build_distil_fl(models: list[str], datasets: list[str],
                    fl_weight: float = 0.1,
                    max_steps: int = -1, **kwargs) -> list[JobNode]:
    """Final-layer hidden-state distillation experiment builder -- all phases."""
    datasets = filter_trainable_datasets(datasets, "distillation_final_layer")
    nodes = []
    for dataset in datasets:
        base_dataset, max_samples = parse_dataset_max(dataset)
        train_ds = get_train_dataset(base_dataset)
        train_nodes_by_model = {}

        hparam_tag = get_hparam_tag(MATCH_JOINT,
                                     final_layer_distill_weight=fl_weight)
        for model in models:
            ms = model_short(model)
            out_dir = get_out_dir(MATCH_JOINT, dataset, model,
                                  hparam_tag=hparam_tag)
            train_name = f"distil_fl_train_{dataset}_{ms}"
            train_nodes_by_model[ms] = train_name
            meta = {"mode": MATCH_JOINT, "dataset": dataset,
                    "model": model, "out_dir": out_dir, "hparam_tag": hparam_tag}

            if find_latest_checkpoint(out_dir):
                nodes.append(JobNode(name=train_name, status="skipped",
                                     phase="train", experiment="distillation_final_layer",
                                     metadata=meta))
            else:
                bsz = get_batch_size(model, MATCH_JOINT)
                cmd = [
                    "python", "-m", "src.train.train_unisd",
                    "--mode", MATCH_JOINT,
                    "--dataset", train_ds,
                    "--model_name", model,
                    "--output_dir", "outputs/unisd",
                    "--max_completion_length", "1024",
                    "--max_prompt_length", "3072",
                    "--per_device_train_batch_size", str(bsz),
                    "--gradient_accumulation_steps", "4",
                    "--final_layer_distill_weight", str(fl_weight),
                    "--use_vllm",
                    "--temperature", "0.7",

                ] + _vllm_gmu_args() + get_dataset_args(train_ds, MATCH_JOINT)

                if max_samples is not None:
                    cmd += ["--num_train_examples", str(max_samples)]
                append_resume_args(cmd, max_steps, kwargs.get("resume_from_checkpoint"),
                                   kwargs.get("additional_epochs", 0))

                nodes.append(JobNode(name=train_name, cmd=cmd, phase="train",
                                     experiment="distillation_final_layer", metadata=meta,
                                     log_file=os.path.join(out_dir, "train.log")))

        nodes += _build_eval_nodes(dataset, MATCH_JOINT, models,
                                   hparam_tag, train_nodes_by_model,
                                   experiment="distillation_final_layer")

    return nodes


def build_mse_fl(models: list[str], datasets: list[str],
                 max_steps: int = -1, **kwargs) -> list[JobNode]:
    """MSE-only final-layer distillation experiment builder -- all phases."""
    datasets = filter_trainable_datasets(datasets, "mse_final_layer")
    nodes = []
    for dataset in datasets:
        base_dataset, max_samples = parse_dataset_max(dataset)
        train_ds = get_train_dataset(base_dataset)
        train_nodes_by_model = {}

        for model in models:
            ms = model_short(model)
            out_dir = get_out_dir(MATCH_REPR, dataset, model)
            train_name = f"mse_fl_train_{dataset}_{ms}"
            train_nodes_by_model[ms] = train_name
            meta = {"mode": MATCH_REPR, "dataset": dataset,
                    "model": model, "out_dir": out_dir, "hparam_tag": ""}

            if find_latest_checkpoint(out_dir):
                nodes.append(JobNode(name=train_name, status="skipped",
                                     phase="train", experiment="mse_final_layer",
                                     metadata=meta))
            else:
                bsz = get_batch_size(model, MATCH_REPR)
                cmd = [
                    "python", "-m", "src.train.train_unisd",
                    "--mode", MATCH_REPR,
                    "--dataset", train_ds,
                    "--model_name", model,
                    "--output_dir", "outputs/unisd",
                    "--max_completion_length", "1024",
                    "--max_prompt_length", "3072",
                    "--per_device_train_batch_size", str(bsz),
                    "--gradient_accumulation_steps", "4",
                    "--use_vllm",
                    "--temperature", "0.7",
                ] + _vllm_gmu_args() + get_dataset_args(train_ds, MATCH_REPR)

                if max_samples is not None:
                    cmd += ["--num_train_examples", str(max_samples)]
                append_resume_args(cmd, max_steps, kwargs.get("resume_from_checkpoint"),
                                   kwargs.get("additional_epochs", 0))

                nodes.append(JobNode(name=train_name, cmd=cmd, phase="train",
                                     experiment="mse_final_layer", metadata=meta,
                                     log_file=os.path.join(out_dir, "train.log")))

        nodes += _build_eval_nodes(dataset, MATCH_REPR, models,
                                   "", train_nodes_by_model,
                                   experiment="mse_final_layer")

    return nodes


def build_clip(models: list[str], datasets: list[str],
               token_clip: float | None = 10.0,
               token_clip_quantile: float | None = None,
               alpha: float = 0.0,
               max_steps: int = -1, **kwargs) -> list[JobNode]:
    """CLIP mode: per-token KL/JSD divergence clipping (the user's `clip` method).

    Either ``token_clip`` (static cap) OR ``token_clip_quantile`` (adaptive
    per-batch quantile cap) may be set, not both. Static default is 10.0, which
    caps only the outlier tail of the per-token loss distribution while preserving
    moderate hard-token signal.
    """
    assert not (token_clip is not None and token_clip_quantile is not None), \
        "Use either token_clip or token_clip_quantile, not both."
    datasets = filter_trainable_datasets(datasets, "clip")
    hparam_tag = get_hparam_tag(CLIP, token_clip=token_clip,
                                token_clip_quantile=token_clip_quantile,
                                alpha=alpha)
    nodes = []
    for dataset in datasets:
        base_dataset, max_samples = parse_dataset_max(dataset)
        train_ds = get_train_dataset(base_dataset)
        train_nodes_by_model = {}

        for model in models:
            ms = model_short(model)
            out_dir = get_out_dir(CLIP, dataset, model, hparam_tag)
            train_name = f"clip_train_{dataset}_{ms}"
            train_nodes_by_model[ms] = train_name
            meta = {"mode": CLIP, "dataset": dataset,
                    "model": model, "out_dir": out_dir, "hparam_tag": hparam_tag}

            if find_latest_checkpoint(out_dir):
                nodes.append(JobNode(name=train_name, status="skipped",
                                     phase="train", experiment="clip",
                                     metadata=meta))
            else:
                bsz = get_batch_size(model, "clip")
                cmd = [
                    "python", "-m", "src.train.train_unisd",
                    "--mode", CLIP,
                    "--dataset", train_ds,
                    "--model_name", model,
                    "--output_dir", "outputs/unisd",
                    "--max_completion_length", "1024",
                    "--max_prompt_length", "3072",
                    "--per_device_train_batch_size", str(bsz),
                    "--gradient_accumulation_steps", "4",
                    "--use_vllm",
                    "--temperature", "0.7",
                    "--alpha", str(alpha),
                ] + _vllm_gmu_args() + get_dataset_args(train_ds, "clip")
                if token_clip_quantile is not None:
                    cmd += ["--token_clip_quantile", str(token_clip_quantile)]
                elif token_clip is not None:
                    cmd += ["--token_clip", str(token_clip)]

                if max_samples is not None:
                    cmd += ["--num_train_examples", str(max_samples)]
                append_resume_args(cmd, max_steps, kwargs.get("resume_from_checkpoint"),
                                   kwargs.get("additional_epochs", 0))

                nodes.append(JobNode(name=train_name, cmd=cmd, phase="train",
                                     experiment="clip", metadata=meta,
                                     log_file=os.path.join(out_dir, "train.log")))

        nodes += _build_eval_nodes(dataset, CLIP, models,
                                   hparam_tag, train_nodes_by_model,
                                   experiment="clip")

    return nodes


def build_all(models: list[str], datasets: list[str],
              fewshot_method: str = RETRIEVAL,
              gamma: float = 0.001, nctx: int = 3, num_demos: int = 3,
              contrastive_weight: float = 0.01, contrastive_margin: float = 0.1,
              sync_steps: int = 10, mixup_beta: float = 0.98,
              fl_weight: float = 0.01,
              max_steps: int = -1, **kwargs) -> list[JobNode]:
    """UNISD_STAR mode: combines fewshot agreement + contrastive + EMA + final-layer distillation."""
    datasets = filter_trainable_datasets(datasets, "all")
    hparam_tag = get_hparam_tag(UNISD_STAR,
                                fewshot_method=fewshot_method,
                                gamma_agreement=gamma,
                                num_auxiliary_contexts=nctx,
                                contrastive_weight=contrastive_weight,
                                contrastive_margin=contrastive_margin,
                                ref_model_sync_steps=sync_steps,
                                ref_model_mixup_beta=mixup_beta,
                                final_layer_distill_weight=fl_weight)
    nodes = []
    for dataset in datasets:
        base_dataset, _ = parse_dataset_max(dataset)
        train_ds = get_train_dataset(base_dataset)
        _, max_samples = parse_dataset_max(dataset)
        train_nodes_by_model = {}

        for model in models:
            ms = model_short(model)

            # Cache nodes: induction cache (if induction method) + neg demo cache
            cache_deps = []

            if fewshot_method == INDUCTION:
                cache_name = f"cache_induction_{train_ds}_{ms}"
                cache_path = get_induction_cache_path("outputs", model, train_ds, nctx, num_demos)
                cache_cmd = [
                    "python", "-m", "src.teacher.instruction_induction",
                    "--dataset", train_ds,
                    "--model_name", model,
                    "--num-auxiliary-contexts", str(nctx),
                    "--induction_num_demos", str(num_demos),
                    "--output_dir", "outputs",
                    "--seed", "42",
                ]
                if cache_path.exists():
                    nodes.append(JobNode(name=cache_name, cmd=cache_cmd, status="skipped",
                                         phase="cache", experiment="all",
                                         metadata={"dataset": train_ds, "model": model}))
                else:
                    nodes.append(JobNode(name=cache_name, cmd=cache_cmd, phase="cache",
                                         experiment="all",
                                         log_file=f"logs/induction_cache_{train_ds}_{ms}.log",
                                         metadata={"dataset": train_ds, "model": model}))
                cache_deps.append(cache_name)

            # Neg demo cache (required for contrastive loss)
            negdemo_name = f"negdemo_{train_ds}_{ms}"
            cache_path = get_neg_demo_cache_path("outputs/cache/demonstration", train_ds, model, None)
            if cache_path.exists():
                nodes.append(JobNode(name=negdemo_name, status="skipped",
                                     phase="cache", experiment="all",
                                     metadata={"dataset": train_ds, "model": model}))
            else:
                cache_cmd = [
                    "python", "-m", "src.teacher.negative_demonstrations",
                    "--model_name", model,
                    "--dataset", train_ds,
                ]
                nodes.append(JobNode(name=negdemo_name, cmd=cache_cmd, phase="cache",
                                     experiment="all",
                                     log_file=f"logs/negdemo_{train_ds}_{ms}.log",
                                     metadata={"dataset": train_ds, "model": model}))
            cache_deps.append(negdemo_name)

            # Train node
            out_dir = get_out_dir(UNISD_STAR, dataset, model, hparam_tag)
            train_name = f"all_train_{dataset}_{ms}"
            train_nodes_by_model[ms] = train_name
            meta = {"mode": UNISD_STAR, "dataset": dataset, "model": model,
                    "out_dir": out_dir, "hparam_tag": hparam_tag}

            if find_latest_checkpoint(out_dir):
                nodes.append(JobNode(name=train_name, status="skipped",
                                     depends_on=cache_deps,
                                     phase="train", experiment="all", metadata=meta))
            else:
                bsz = get_batch_size(model, "all")
                # UNISD_STAR mode is memory-heavy: halve batch size
                bsz = max(1, bsz // 2)
                mode_args = [
                    "--fewshot_method", fewshot_method,
                    "--num-auxiliary-contexts", str(nctx),
                    "--gamma_agreement", str(gamma),
                    "--contrastive_weight", str(contrastive_weight),
                    "--contrastive_margin", str(contrastive_margin),
                    "--ref_model_sync_steps", str(sync_steps),
                    "--ref_model_mixup_beta", str(mixup_beta),
                    "--final_layer_distill_weight", str(fl_weight),
                ]
                if fewshot_method == INDUCTION:
                    mode_args += ["--induction_num_demos", str(num_demos)]

                cmd = [
                    "python", "-m", "src.train.train_unisd",
                    "--mode", UNISD_STAR,
                    "--dataset", train_ds,
                    "--model_name", model,
                    "--output_dir", "outputs/unisd",
                    "--max_completion_length", "1024",
                    "--max_prompt_length", "3072",
                    "--per_device_train_batch_size", str(bsz),
                    "--gradient_accumulation_steps", "4",
                    "--use_vllm",
                    "--temperature", "0.7",
                ] + _vllm_gmu_args() + mode_args + get_dataset_args(train_ds, "all")

                if max_samples is not None:
                    cmd += ["--num_train_examples", str(max_samples)]
                append_resume_args(cmd, max_steps, kwargs.get("resume_from_checkpoint"),
                                   kwargs.get("additional_epochs", 0))

                nodes.append(JobNode(name=train_name, cmd=cmd,
                                     depends_on=cache_deps,
                                     phase="train", experiment="all", metadata=meta,
                                     log_file=os.path.join(out_dir, "train.log")))

        nodes += _build_eval_nodes(dataset, UNISD_STAR, models, hparam_tag,
                                   train_nodes_by_model, experiment="all")

    return nodes


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════


def parse_args():
    # Common options shared by all subcommands
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--gpus", type=str, default="0,1,2,3,4,5",
                        help="Comma-separated GPU IDs (default: 0,1,2,3,4)")
    common.add_argument("--models", type=str, default=None,
                        help="Comma-separated model names (default: 4 Qwen models)")
    common.add_argument("--datasets", type=str, default=None,
                        help="Comma-separated dataset names")
    common.add_argument("--dry-run", action="store_true",
                        help="Print jobs without executing")
    common.add_argument("--max-steps", type=int, default=-1,
                        help="Limit training steps (for smoke testing)")
    common.add_argument("--resume-from-checkpoint", type=str, default=None,
                        help="Path to checkpoint dir to resume training from")
    common.add_argument("--additional-epochs", type=float, default=0,
                        help="Additional epochs to train beyond checkpoint (e.g. 0.2)")
    common.add_argument("--base-port", type=int, default=29500,
                        help="Base MASTER_PORT (default: 29500)")
    common.add_argument("--gpu-memory-utilization", type=float, default=None,
                        help="Override vLLM memory fraction. Used for eval "
                             "(--gpu_memory_utilization) and training "
                             "(--vllm_gpu_memory_utilization). E.g. 0.5 for half-GPU.")
    common.add_argument("--no-vllm-sleep", action="store_true",
                        help="Pass --no-vllm_enable_sleep_mode to every training "
                             "command. Required on drivers/CUDA combos where "
                             "vllm/device_allocator/cumem.py raises 'CUDA driver "
                             "version is insufficient for CUDA runtime version'.")

    parser = argparse.ArgumentParser(
        description="Unified experiment runner with dependency-aware GPU scheduler.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[common],
    )
    sub = parser.add_subparsers(dest="experiment")
    sub.required = False
    sub.default = None

    # induction
    p = sub.add_parser(INDUCTION, parents=[common], help="Induction-based distillation")
    p.add_argument("--gamma", type=float, default=0.01)
    p.add_argument("--nctx", type=int, default=3)
    p.add_argument("--num-demos", type=int, default=3)

    # agreement (multi-mode teacher-agreement)
    p = sub.add_parser(AGREEMENT, parents=[common], help="Multi-mode teacher agreement")
    p.add_argument("--modes", type=str,
                   default=",".join([AGREEMENT_SEQ_RANDOM, AGREEMENT_SEQ_RETRIEVAL, AGREEMENT_SEQ_INDUCTION,
                                     AGREEMENT_TOK_RANDOM, AGREEMENT_TOK_RETRIEVAL, AGREEMENT_TOK_INDUCTION]),
                   help="Comma-separated training modes")
    p.add_argument("--gamma", type=float, default=0.01)
    p.add_argument("--nctx", type=int, default=3)
    p.add_argument("--num-demos", type=int, default=3)

    # ema (EMA-teacher distillation)
    p = sub.add_parser(EMA, parents=[common], help="EMA teacher distillation")
    p.add_argument("--sync-steps", type=int, default=10)
    p.add_argument("--mixup-beta", type=float, default=0.9)

    # contrastive (token-level contrastive learning)
    p = sub.add_parser(CONTRASTIVE, parents=[common], help="Token contrastive learning")
    p.add_argument("--weight", type=float, default=0.1)
    p.add_argument("--margin", type=float, default=0.5)

    # match_joint (final-layer hidden-state distillation + KL)
    p = sub.add_parser(MATCH_JOINT, parents=[common], help="Final-layer hidden-state distillation")
    p.add_argument("--fl-weight", type=float, default=0.1,
                   help="Final-layer distillation weight (default: 0.1)")

    # match_repr (MSE-only final-layer distillation, no KL)
    sub.add_parser(MATCH_REPR, parents=[common], help="MSE-only final-layer distillation (no KL)")

    # clip (per-token KL/JSD divergence clipping)
    p = sub.add_parser(CLIP, parents=[common], help="Per-token KL/JSD clipping")
    p.add_argument("--token-clip", type=float, default=None,
                   help="Per-token KL/JSD clip ceiling (build_clip default: 10.0). "
                        "Mutually exclusive with --token-clip-quantile.")
    p.add_argument("--token-clip-quantile", type=float, default=None,
                   help="Adaptive per-batch quantile clip in (0, 1]. "
                        "Mutually exclusive with --token-clip.")
    p.add_argument("--alpha", type=float, default=0.0,
                   help="JSD interpolation: 0.0=forward KL, 1.0=reverse KL (default: 0.0)")

    # unisd_star (combined: fewshot agreement + contrastive + EMA + final-layer distillation)
    p = sub.add_parser(UNISD_STAR, parents=[common], help="UniSD* combined recipe")
    p.add_argument("--fewshot-method", type=str, default=RETRIEVAL,
                   choices=sorted(FEWSHOT_METHODS),
                   help=f"Fewshot method (default: {RETRIEVAL})")
    p.add_argument("--gamma", type=float, default=0.001)
    p.add_argument("--nctx", type=int, default=3)
    p.add_argument("--num-demos", type=int, default=3)
    p.add_argument("--contrastive-weight", type=float, default=0.01)
    p.add_argument("--contrastive-margin", type=float, default=0.1)
    p.add_argument("--sync-steps", type=int, default=10)
    p.add_argument("--mixup-beta", type=float, default=0.98)
    p.add_argument("--fl-weight", type=float, default=0.01)

    return parser.parse_args()


def _dispatch(experiment: str, args) -> tuple:
    """Return (build_fn, extra_kwargs) for a given experiment name.

    Subcommand names mirror the canonical constants in src/const.py
    (AGREEMENT, EMA, CONTRASTIVE, MATCH_JOINT, MATCH_REPR, CLIP, UNISD_STAR, INDUCTION).
    """
    if experiment == INDUCTION:
        gamma = getattr(args, "gamma", 0.01)
        nctx = getattr(args, "nctx", 3)
        num_demos = getattr(args, "num_demos", 3)
        return build_induction, {"gamma": gamma, "nctx": nctx, "num_demos": num_demos}
    elif experiment == AGREEMENT:
        gamma = getattr(args, "gamma", 0.01)
        nctx = getattr(args, "nctx", 3)
        num_demos = getattr(args, "num_demos", 3)
        modes = getattr(args, "modes", ",".join([AGREEMENT_SEQ_RANDOM, AGREEMENT_SEQ_RETRIEVAL, AGREEMENT_SEQ_INDUCTION,
                                                  AGREEMENT_TOK_RANDOM, AGREEMENT_TOK_RETRIEVAL,
                                                  AGREEMENT_TOK_INDUCTION]))
        return build_teacher_agreement, {
            "modes": modes.split(",") if isinstance(modes, str) else modes,
            "gamma": gamma,
            "nctx": nctx,
            "num_demos": num_demos,
        }
    elif experiment == EMA:
        sync_steps = getattr(args, "sync_steps", 10)
        mixup_beta = getattr(args, "mixup_beta", 0.9)
        return build_teacher_ema, {"sync_steps": sync_steps, "mixup_beta": mixup_beta}
    elif experiment == CONTRASTIVE:
        weight = getattr(args, "weight", 0.1)
        margin = getattr(args, "margin", 0.5)
        return build_token_cl, {"weight": weight, "margin": margin}
    elif experiment == MATCH_JOINT:
        fl_weight = getattr(args, "fl_weight", 0.1)
        return build_distil_fl, {"fl_weight": fl_weight}
    elif experiment == MATCH_REPR:
        return build_mse_fl, {}
    elif experiment == CLIP:
        tc = getattr(args, "token_clip", None)
        tcq = getattr(args, "token_clip_quantile", None)
        if tc is None and tcq is None:
            tc = 10.0  # matches train_args.py default
        return build_clip, {
            "token_clip": tc,
            "token_clip_quantile": tcq,
            "alpha": getattr(args, "alpha", 0.0),
        }
    elif experiment == UNISD_STAR:
        return build_all, {
            "fewshot_method": getattr(args, "fewshot_method", RETRIEVAL),
            "gamma": getattr(args, "gamma", 0.001),
            "nctx": getattr(args, "nctx", 3),
            "num_demos": getattr(args, "num_demos", 3),
            "contrastive_weight": getattr(args, "contrastive_weight", 0.01),
            "contrastive_margin": getattr(args, "contrastive_margin", 0.1),
            "sync_steps": getattr(args, "sync_steps", 10),
            "mixup_beta": getattr(args, "mixup_beta", 0.98),
            "fl_weight": getattr(args, "fl_weight", 0.01),
        }
    else:
        print(f"Unknown experiment: {experiment}")
        sys.exit(1)


def main():
    args = parse_args()

    global _GPU_MEM_UTIL, _DISABLE_VLLM_SLEEP
    _GPU_MEM_UTIL = args.gpu_memory_utilization
    _DISABLE_VLLM_SLEEP = bool(getattr(args, "no_vllm_sleep", False))

    gpu_ids = [int(g) for g in args.gpus.split(",")]
    models = args.models.split(",") if args.models else DEFAULT_MODELS

    scheduler = DependencyScheduler(gpu_ids=gpu_ids, base_port=args.base_port)
    os.makedirs("logs", exist_ok=True)

    experiments = [args.experiment] if args.experiment else ALL_EXPERIMENTS

    print(f"GPUs: {gpu_ids}")
    print(f"Models: {[model_short(m) for m in models]}")
    if args.max_steps > 0:
        print(f"Max steps: {args.max_steps}")
    if args.dry_run:
        print("DRY RUN -- no jobs will be executed")

    # Materialize UNISD_STAR jobs from UNISD_STAR experiments into a single graph
    graph = JobGraph()
    for experiment in experiments:
        datasets = args.datasets.split(",") if args.datasets else DEFAULT_DATASETS
        build_fn, extra = _dispatch(experiment, args)
        nodes = build_fn(models=models, datasets=datasets, max_steps=args.max_steps,
                         resume_from_checkpoint=args.resume_from_checkpoint,
                         additional_epochs=args.additional_epochs, **extra)
        graph.add_all(nodes)

    if args.dry_run:
        scheduler.run_dry(graph)
    else:
        failed = scheduler.run(graph)
        if not failed:
            print("\nRun: python summarize_results.py")


if __name__ == "__main__":
    main()
