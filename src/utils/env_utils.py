"""Centralized environment-variable helpers for the UniSD project.

Reads MODEL_DTYPE from the project-root .env file (no python-dotenv dependency).
os.environ takes precedence over the file value.

Calling :func:`load_project_env` propagates *all* ``.env`` vars into
``os.environ`` (without overwriting existing vars).  This is important for
variables like ``VLLM_ATTENTION_BACKEND`` that must be set before vLLM is
imported.

Importing this module also patches ``transformers.utils`` to expose
``LossKwargs`` on builds that pre-date the symbol — required for custom-code
models (e.g. InternLM3) to load. Modules that previously imported
``src.utils.compat`` for this side effect can now rely on importing
``env_utils`` instead (or any symbol from it).
"""

import os
from pathlib import Path
from typing import TypedDict

import transformers.utils

_project_env_loaded = False


# --- transformers compatibility shim --------------------------------------- #
# Patch must run at module import time, before any HF model loading.

if not hasattr(transformers.utils, "LossKwargs"):

    class LossKwargs(TypedDict, total=False):
        labels: "torch.LongTensor"  # noqa: F821

    transformers.utils.LossKwargs = LossKwargs
else:
    LossKwargs = transformers.utils.LossKwargs  # type: ignore[assignment]

_DTYPE_MAP = {
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "fp16": "float16",
    "float16": "float16",
    "fp32": "float32",
    "float32": "float32",
    "auto": "auto",
}

_DEFAULT_DTYPE = "bfloat16"


def _load_dotenv_value(key: str) -> str | None:
    """Parse the project-root .env file and return the value for *key*, or None."""
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.is_file():
        return None
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, _, v = line.partition("=")
            if k.strip() == key:
                return v.strip()
    return None


def load_project_env() -> None:
    """Load all variables from the project-root ``.env`` into ``os.environ``.

    Already-set environment variables are **not** overwritten, so explicit
    exports / launch.json ``env`` blocks always win.  The function is
    idempotent — repeated calls are no-ops.
    """
    global _project_env_loaded
    if _project_env_loaded:
        return
    _project_env_loaded = True

    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.is_file():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip()
            if k and k not in os.environ:
                os.environ[k] = v


def get_model_dtype() -> str:
    """Return the vLLM-compatible dtype string (e.g. ``"bfloat16"``, ``"float16"``).

    Resolution order:
      1. ``os.environ["MODEL_DTYPE"]`` (set by shell / launcher)
      2. ``MODEL_DTYPE`` in ``<project_root>/.env``
      3. Default: ``"bfloat16"``
    """
    load_project_env()  # ensure .env vars (including VLLM_ATTENTION_BACKEND) are in os.environ
    raw = os.environ.get("MODEL_DTYPE") or _load_dotenv_value("MODEL_DTYPE") or ""
    raw = raw.strip().lower()
    return _DTYPE_MAP.get(raw, _DEFAULT_DTYPE)


def get_torch_dtype():
    """Return a ``torch.dtype`` matching :func:`get_model_dtype`.

    Useful for non-vLLM code paths (e.g. LoRA merge).
    """
    import torch

    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype_str = get_model_dtype()
    return mapping.get(dtype_str, torch.bfloat16)


def disable_thinking_mode(tokenizer):
    """Disable thinking mode for Qwen3+ models by modifying chat template."""
    if (hasattr(tokenizer, 'chat_template')
        and tokenizer.chat_template
        and 'enable_thinking' in tokenizer.chat_template):
        tokenizer.chat_template = tokenizer.chat_template.replace(
            'enable_thinking is defined and enable_thinking is false',
            'true',
        )
        print("[Tokenizer] Disabled thinking mode")
    return tokenizer
