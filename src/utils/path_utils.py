import os
from pathlib import Path

from src.const import (CONTRASTIVE, EMA, MATCH_JOINT, MATCH_REPR, UNISD_STAR,
                       CLIP, RANDOM)


def build_hparam_tag(mode: str, **kwargs) -> str:
    """Canonical hparam subdirectory tag. Empty string → no extra dir level."""
    if mode == MATCH_REPR:
        return ""
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
    # all fewshot/induction modes (including _token variants)
    return f"gamma{kwargs['gamma_agreement']}_nctx{kwargs['num_auxiliary_contexts']}"


def get_induction_cache_path(output_dir: str, model_name: str, dataset: str, num_auxiliary_contexts: int, num_demos: int) -> Path:
    induction_cache_file = (
        Path("outputs") / "cache" / "induction" / dataset / os.path.basename(model_name) /
        f"{os.path.basename(model_name)}_induction-random"
        f"_{num_auxiliary_contexts}-contexts_{num_demos}-demos.json"
    )
    return induction_cache_file


def get_neg_demo_cache_path(cache_dir: str, dataset_name: str, model_name: str, max_samples: str) -> Path:
    suffix = ""
    if max_samples is not None:
        suffix += f"_max{max_samples}"

    path = Path(cache_dir) / dataset_name / os.path.basename(model_name) / f"neg_demos{suffix}.json"
    return path


def get_model_dir(output_dir: str, method: str, dataset: str, model_name: str, subset: str = None, hparam_tag: str = "", max_samples: int | None = None) -> str:
    if subset:
        dataset_tag = subset if subset.startswith(dataset) else f"{dataset}-{subset}"
    else:
        dataset_tag = dataset
    if max_samples is not None:
        dataset_tag += f"-max{max_samples}"
    parts = [output_dir, method, dataset_tag]
    if hparam_tag:
        parts.append(hparam_tag)
    parts.append(os.path.basename(model_name))
    model_path = Path(*parts)
    return model_path


def get_eval_results_path(output_dir: str, mode: str, dataset: str, model_name: str, split: str, file_type: str, subset: str = None, hparam_tag: str = "", suffix: str = "") -> str:
    """Build eval results path: {output_dir}/{mode}/{dataset}/[{hparam_tag}/]{model}/results_...{ext}

    When model_name is a checkpoint path (e.g. .../Qwen2.5-7B-Instruct/checkpoint-406),
    uses the parent directory name (the model name) instead of the checkpoint name.
    """
    model_path = Path(model_name)
    if model_path.name.startswith("checkpoint-") or model_path.name in ("_merged", "final"):
        model_short = model_path.parent.name
        # Handle _merged inside checkpoint: .../Qwen2.5-7B/checkpoint-406/_merged
        if model_short.startswith("checkpoint-") or model_short == "final":
            model_short = model_path.parent.parent.name
    else:
        model_short = model_path.name
    # Avoid duplication like "gpqa-gpqa_main" — if subset already starts with dataset name, use subset alone
    if subset:
        dataset_tag = subset if subset.startswith(dataset) else f"{dataset}-{subset}"
    else:
        dataset_tag = dataset
    if suffix:
        dataset_tag += f"-{suffix}"

    filename = f"results_{dataset_tag}_{split}_{model_short}.{file_type}"
    dir_dataset = dataset_tag
    parts = [output_dir, mode, dir_dataset]
    if hparam_tag:
        parts.append(hparam_tag)
    parts.extend([model_short, filename])
    path = Path(*parts)
    return path
