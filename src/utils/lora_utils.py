"""LoRA utilities for training and inference.

Provides functions to detect LoRA checkpoints, load+merge adapters,
prepare merged models for vLLM, and get the default LoRA config.
"""

import json
import os
import logging
import shutil

logger = logging.getLogger(__name__)


def is_lora_checkpoint(path: str) -> bool:
    """Check if a checkpoint directory contains a LoRA adapter."""
    return os.path.isfile(os.path.join(path, "adapter_config.json"))


def get_base_model_path(adapter_path: str) -> str:
    """Read the base model path from a LoRA adapter config."""
    config_path = os.path.join(adapter_path, "adapter_config.json")
    with open(config_path) as f:
        config = json.load(f)
    return config["base_model_name_or_path"]


def load_model_for_inference(path: str, torch_dtype=None, device_map="auto"):
    """Load a model for inference, auto-merging LoRA if detected.

    If path contains adapter_config.json, loads the base model and merges
    the adapter. Otherwise loads the model directly (backward compat).

    Returns:
        A merged model ready for inference.
    """
    from transformers import AutoModelForCausalLM

    if is_lora_checkpoint(path):
        from peft import PeftModel

        base_path = get_base_model_path(path)
        logger.info("Loading base model from %s for LoRA merge", base_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_path, torch_dtype=torch_dtype, device_map=device_map, trust_remote_code=True,
        )
        logger.info("Loading LoRA adapter from %s", path)
        model = PeftModel.from_pretrained(base_model, path)
        model = model.merge_and_unload()
        logger.info("LoRA adapter merged successfully")
        return model
    else:
        return AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch_dtype, device_map=device_map, trust_remote_code=True,
        )


def _resolve_checkpoint_path(path: str) -> str:
    """Resolve an output directory to its latest checkpoint if needed.

    If path is not a local directory (e.g. a HuggingFace model ID), return
    it unchanged. If path is already a model dir (has config.json or
    adapter_config.json), return it unchanged. Otherwise, find the latest
    checkpoint-N subdirectory.
    """
    if not os.path.isdir(path):
        return path
    if os.path.isfile(os.path.join(path, "config.json")) or is_lora_checkpoint(path):
        return path

    # Look for checkpoint-N subdirectories
    checkpoints = sorted(
        [d for d in os.listdir(path)
         if os.path.isdir(os.path.join(path, d)) and d.startswith("checkpoint-")],
        key=lambda d: int(d.split("-")[-1]),
    )
    if checkpoints:
        resolved = os.path.join(path, checkpoints[-1])
        logger.info("Resolved output dir to latest checkpoint: %s", resolved)
        return resolved

    return path


def prepare_model_for_vllm(path: str, torch_dtype=None, force_remerge: bool = False) -> str:
    """Prepare a checkpoint for vLLM by merging LoRA if needed.

    If the checkpoint is a LoRA adapter:
      1. Check for cached merge at <path>/_merged/
      2. If not cached (or force_remerge), load base + adapter on CPU,
         merge, and save to <path>/_merged/
      3. Return the merged path

    If not a LoRA checkpoint, return path unchanged (backward compat).

    Returns:
        Path to a full model directory suitable for vLLM.
    """
    path = _resolve_checkpoint_path(path)

    if not is_lora_checkpoint(path):
        return path

    merged_path = os.path.join(path, "_merged")

    # Cache check
    if os.path.isfile(os.path.join(merged_path, "config.json")) and not force_remerge:
        logger.info("Using cached merged model at %s", merged_path)
        return merged_path

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    if torch_dtype is None:
        from src.utils.env_utils import get_torch_dtype
        torch_dtype = get_torch_dtype()

    base_path = get_base_model_path(path)
    logger.info("Merging LoRA adapter for vLLM: base=%s, adapter=%s", base_path, path)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_path, torch_dtype=torch_dtype, device_map="cpu", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, path, device_map="cpu")
    model = model.merge_and_unload()

    logger.info("Saving merged model to %s", merged_path)
    os.makedirs(merged_path, exist_ok=True)
    model.save_pretrained(merged_path)

    # Save tokenizer so vLLM can load from the same dir
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    tokenizer.save_pretrained(merged_path)

    # Copy processor files for multimodal models (e.g. Gemma 3)
    from transformers.utils import cached_file
    for proc_file in ("preprocessor_config.json", "processor_config.json"):
        try:
            resolved = cached_file(base_path, proc_file)
            if resolved and os.path.isfile(resolved):
                dest = os.path.join(merged_path, proc_file)
                if not os.path.isfile(dest):
                    shutil.copy2(resolved, dest)
                    logger.info("Copied %s from base model", proc_file)
        except Exception:
            pass  # File doesn't exist in base model, skip

    # Free CPU/GPU memory from merge before vLLM takes over
    del model, base_model, tokenizer
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Merged model saved (%s)", merged_path)
    return merged_path


def get_default_lora_config(model_name: str = None):
    """Return the default LoRA config, with model-aware target modules.

    Phi-4 uses fused projections (qkv_proj, gate_up_proj) while
    Qwen, Gemma, and most other models use separate projections.
    """
    from peft import LoraConfig, TaskType

    if model_name and "phi" in model_name.lower():
        target_modules = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]
    else:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

    return LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
