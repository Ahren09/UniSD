"""Extraction and scoring utilities for MCQA evaluation (GPQA, ScienceQA, etc.)."""

import contextlib
import glob
import json
import re
import signal
import sys
from collections import defaultdict
import random
import numpy as np
import logging
import warnings
import pandas as pd
from tqdm import trange

def _json_candidates(text: str) -> list[str]:
    """Yield candidate JSON strings: raw text, then content of fenced blocks."""
    candidates = [text]
    # Extract ```json ... ``` or ``` ... ``` fenced blocks
    for m in re.finditer(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL):
        candidates.append(m.group(1).strip())
    return candidates


def set_seed(seed: int = 42, use_torch: bool = True):
    random.seed(seed)
    np.random.seed(seed)

    if use_torch:
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.enabled = False
        except ImportError:
            print("Fail to import torch. Skipping torch seed setting")



def project_setup():
    
    warnings.simplefilter(action='ignore', category=FutureWarning)
    pd.set_option('display.max_rows', 40)
    pd.set_option('display.max_columns', 20)
    set_seed(42)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)
    logging.getLogger("google").setLevel(logging.WARNING)


def extract_answer(text: str, structured_output: str | None = None, num_choices: int = 4) -> str | None:
    """Extract the answer letter from model output.

    Args:
        text: Raw model output.
        structured_output: None for free-form, "letter", or "json".
        num_choices: Number of choices (2-5). Determines valid letters.

    Returns:
        Single letter (A-E), or None on extraction failure.
    """
    text = text.strip()
    if not text:
        return None

    valid_letters = "".join(chr(ord('A') + i) for i in range(num_choices))
    letter_pattern = f"[{valid_letters}]"

    if structured_output == "letter":
        ch = text[0].upper()
        return ch if ch in valid_letters else None

    if structured_output == "json":
        # Try json.loads on raw text first
        for candidate in _json_candidates(text):
            try:
                parsed = json.loads(candidate)
                ans = parsed.get("answer", "").strip().upper()
                if ans and ans[0] in valid_letters:
                    return ans[0]
            except (json.JSONDecodeError, ValueError, AttributeError):
                continue
        # Fallback: regex for "answer": "X"
        m = re.search(rf'"answer"\s*:\s*"({letter_pattern})"', text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
        return None

    # Free-form CoT extraction
    # 1. "Answer: X" (matches the prompt format)
    m = re.search(rf"^Answer:\s*\(?({letter_pattern})\)?", text, re.MULTILINE)
    if m:
        return m.group(1)

    # 2. "The answer is X"
    m = re.search(rf"[Tt]he answer is\s*\(?({letter_pattern})\)?", text)
    if m:
        return m.group(1)

    # 3. "ANSWER: X"
    m = re.search(rf"ANSWER:\s*\(?({letter_pattern})\)?", text)
    if m:
        return m.group(1)

    # 4. Line starting with a letter
    m = re.search(rf"^({letter_pattern})[\.\)\s]", text, re.MULTILINE)
    if m:
        return m.group(1)

    # 5. Last standalone letter as final fallback
    matches = re.findall(rf"\b({letter_pattern})\b", text)
    if matches:
        return matches[-1]

    return None


def compute_accuracy(results: list[dict]) -> float:
    """Compute overall accuracy from result dicts with 'correct' bool key."""
    if not results:
        return 0.0
    return sum(r["correct"] for r in results) / len(results)


def compute_per_domain_accuracy(results: list[dict]) -> dict[str, float]:
    """Compute accuracy breakdown by domain."""
    domain_correct = defaultdict(list)
    for r in results:
        domain = r.get("domain", "unknown")
        domain_correct[domain].append(r["correct"])
    return {
        domain: sum(vals) / len(vals)
        for domain, vals in sorted(domain_correct.items())
    }


def compute_extraction_success_rate(results: list[dict]) -> float:
    """Fraction of results where answer extraction succeeded."""
    if not results:
        return 0.0
    return sum(1 for r in results if r.get("extracted_answer") is not None) / len(results)


def _pretokenize_prompts_and_completions(
    prompts: list[str],
    completions: list[str],
    tokenizer,
    max_length: int | None,
) -> list[tuple[list[int], list[int]]]:
    """Tokenize each (prompt, completion) pair separately and enforce a joint length budget.

    Both strings are encoded with ``add_special_tokens=False`` — chat-template / BOS
    tokens are assumed to already be baked into the prompt strings (this matches the
    way vLLM was called in the eval scripts).

    Length-budget policy: when ``len(prompt_ids) + len(completion_ids) > max_length``,
    the PROMPT is front-truncated and the completion is kept intact. This is the only
    correct choice for retention-style scoring: truncating the completion would change
    which tokens we score. If the completion alone exceeds the budget, the completion
    tail is kept (the front is dropped) and the prompt is zeroed out. A single warning
    is printed on the first truncation event per call.

    Returns:
        A list the same length as ``prompts`` of ``(prompt_ids, completion_ids)``
        tuples. Entries for empty completions have ``completion_ids == []``; callers
        are expected to skip those in the forward pass.
    """
    assert len(prompts) == len(completions), (
        f"prompts and completions must have the same length, got {len(prompts)} vs {len(completions)}"
    )

    pre_tokenized: list[tuple[list[int], list[int]]] = []
    truncation_warned = False
    for p, c in zip(prompts, completions):
        prompt_ids = tokenizer.encode(p, add_special_tokens=False)
        completion_ids = tokenizer.encode(c, add_special_tokens=False) if c else []
        if max_length is not None and len(prompt_ids) + len(completion_ids) > max_length:
            keep = max_length - len(completion_ids)
            if keep <= 0:
                if not truncation_warned:
                    print(
                        f"[_pretokenize_prompts_and_completions] WARNING: a completion of "
                        f"{len(completion_ids)} tokens exceeds max_length={max_length}; "
                        f"truncating completion tail (further warnings suppressed)."
                    )
                    truncation_warned = True
                prompt_ids = []
                completion_ids = completion_ids[-max_length:]
            else:
                if not truncation_warned:
                    print(
                        f"[_pretokenize_prompts_and_completions] WARNING: prompt+completion "
                        f"{len(prompt_ids) + len(completion_ids)} > max_length={max_length}; "
                        f"front-truncating prompt (further warnings suppressed)."
                    )
                    truncation_warned = True
                prompt_ids = prompt_ids[-keep:]
        pre_tokenized.append((prompt_ids, completion_ids))
    return pre_tokenized


def score_completions_under_model(
    model,
    tokenizer,
    prompts: list[str],
    completions: list[str],
    batch_size: int = 16,
    max_length: int | None = None,
) -> list[dict]:
    """Compute log p_model(completion | prompt) per example, average over completion tokens.

    Trainer-agnostic teacher-forcing scorer used by the retention-to-base eval mode.
    Mirrors the math in src/trainers/base_trainer.py:_get_per_token_logps_and_entropies
    but is a simple inference helper that does not depend on any trainer class.

    Args:
        model: An HF CausalLM, already in eval() mode, loaded on a single device.
        tokenizer: The matching tokenizer (loaded from the BASE model path so the
            vocabulary matches the model used for scoring).
        prompts: Already-chat-templated prompt strings (the same strings fed to vLLM
            during generation). Tokenized with add_special_tokens=False since the chat
            template already injects BOS / system tokens.
        completions: Raw generated text from vLLM, one per prompt.
        batch_size: Mini-batch size for the forward pass.
        max_length: Optional truncation budget for prompt+completion. When the sum
            exceeds this, the PROMPT is front-truncated and the completion is kept
            intact (the completion is what we score).

    Returns:
        A list of per-example dicts, in the same order as the inputs:
            {"logp_sum": float, "logp_mean": float, "n_tokens": int}
        Empty completions yield n_tokens=0, logp_sum=0.0, logp_mean=float("nan").
    """
    import torch
    import torch.nn.functional as F

    assert len(prompts) == len(completions), (
        f"prompts and completions must have the same length, got {len(prompts)} vs {len(completions)}"
    )

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError(
            "tokenizer has neither pad_token_id nor eos_token_id; cannot pad batches "
            "for retention scoring."
        )

    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    use_autocast = (device.type == "cuda" and model_dtype in (torch.bfloat16, torch.float16))

    # Pre-tokenize all prompts and completions individually so the completion
    # boundary is unambiguous and we don't lose tokens to BPE-merge boundaries.
    pre_tokenized = _pretokenize_prompts_and_completions(
        prompts, completions, tokenizer, max_length
    )

    results: list[dict] = [None] * len(prompts)  # type: ignore[list-item]

    for batch_start in trange(0, len(pre_tokenized), batch_size):
        batch = pre_tokenized[batch_start : batch_start + batch_size]
        # Track empties; they don't need a forward pass.
        nonempty_indices: list[int] = []  # indices into `batch`
        nonempty_inputs: list[list[int]] = []
        nonempty_completion_lengths: list[int] = []
        for j, (p_ids, c_ids) in enumerate(batch):
            global_idx = batch_start + j
            if len(c_ids) == 0:
                results[global_idx] = {
                    "logp_sum": 0.0,
                    "logp_mean": float("nan"),
                    "n_tokens": 0,
                }
                continue
            nonempty_indices.append(j)
            nonempty_inputs.append(p_ids + c_ids)
            nonempty_completion_lengths.append(len(c_ids))

        if not nonempty_inputs:
            continue

        max_len = max(len(seq) for seq in nonempty_inputs)
        b = len(nonempty_inputs)
        input_ids = torch.full((b, max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((b, max_len), dtype=torch.long)
        completion_mask = torch.zeros((b, max_len), dtype=torch.long)
        for k, seq in enumerate(nonempty_inputs):
            seq_len = len(seq)
            input_ids[k, :seq_len] = torch.tensor(seq, dtype=torch.long)
            attention_mask[k, :seq_len] = 1
            c_len = nonempty_completion_lengths[k]
            completion_mask[k, seq_len - c_len : seq_len] = 1

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        completion_mask = completion_mask.to(device)

        with torch.no_grad():
            if use_autocast:
                with torch.autocast(device_type="cuda", dtype=model_dtype):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # (b, L, V)
            # Standard causal-LM shift: logits at position t predict token t+1.
            shift_logits = logits[:, :-1, :].float()
            shift_labels = input_ids[:, 1:]
            shift_completion_mask = completion_mask[:, 1:].float()
            shift_attention_mask = attention_mask[:, 1:].float()

            log_probs = F.log_softmax(shift_logits, dim=-1)
            token_logps = log_probs.gather(
                dim=-1, index=shift_labels.unsqueeze(-1)
            ).squeeze(-1)  # (b, L-1)

            mask = shift_completion_mask * shift_attention_mask  # (b, L-1)
            per_example_sum = (token_logps * mask).sum(dim=-1)  # (b,)
            per_example_n = mask.sum(dim=-1)                     # (b,)

        per_example_sum_cpu = per_example_sum.detach().cpu().tolist()
        per_example_n_cpu = per_example_n.detach().cpu().tolist()

        for k, j in enumerate(nonempty_indices):
            global_idx = batch_start + j
            n_tokens = int(per_example_n_cpu[k])
            logp_sum = float(per_example_sum_cpu[k])
            logp_mean = (logp_sum / n_tokens) if n_tokens > 0 else float("nan")
            results[global_idx] = {
                "logp_sum": logp_sum,
                "logp_mean": logp_mean,
                "n_tokens": n_tokens,
            }

    # Sanity: every slot is filled.
    assert all(r is not None for r in results), "score_completions_under_model: missing results"
    return results  # type: ignore[return-value]


def _assert_tokenizers_compatible_for_jsd(
    base_tokenizer,
    eval_tokenizer,
    sample_strings: list[str],
) -> None:
    """Defensive check that two tokenizers agree on vocab size and on a small
    sample of encodings. If they disagree, token-level JSD is undefined — raise
    a clean ``ValueError`` instead of silently comparing misaligned distributions.
    """
    base_vocab = getattr(base_tokenizer, "vocab_size", None)
    eval_vocab = getattr(eval_tokenizer, "vocab_size", None)
    if base_vocab is None or eval_vocab is None or base_vocab != eval_vocab:
        raise ValueError(
            "score_completions_jsd_between_models: base and eval tokenizers "
            f"have different vocab sizes (base={base_vocab}, eval={eval_vocab}); "
            "token-level JSD is only defined when the two models share a vocabulary."
        )
    # Sample-encoding check: compare IDs on a small subset of strings.
    for s in sample_strings[: min(8, len(sample_strings))]:
        if not s:
            continue
        base_ids = base_tokenizer.encode(s, add_special_tokens=False)
        eval_ids = eval_tokenizer.encode(s, add_special_tokens=False)
        if base_ids != eval_ids:
            raise ValueError(
                "score_completions_jsd_between_models: base and eval tokenizers "
                "produced different IDs on a sample string; token-level JSD is "
                "only defined when the two models share a vocabulary. "
                f"(first-mismatch sample len: base={len(base_ids)}, eval={len(eval_ids)})"
            )


def score_completions_jsd_between_models(
    base_model,
    base_tokenizer,
    eval_model,
    eval_tokenizer,
    prompts: list[str],
    completions: list[str],
    batch_size: int = 1,
    max_length: int | None = None,
    jsd_log_clamp: float = 1e-12,
) -> list[dict]:
    """Token-level symmetric Jensen-Shannon divergence between the BASE model and
    the EVAL model on FT-generated completions, teacher-forced on the same prefixes.

    For each completion token position ``t`` in each example, the helper forms the
    two next-token distributions::

        p = softmax(base_model(prompt + completion[:t]))       # (V,)
        q = softmax(eval_model(prompt + completion[:t]))       # (V,)
        m = 0.5 * (p + q)
        jsd_t = 0.5 * (KL(p || m) + KL(q || m))                # in [0, log 2] nats

    and averages ``jsd_t`` over the completion tokens of each example. This is
    token-level JSD on a fixed generated completion — it is NOT sequence-level
    Monte-Carlo KL, and it does NOT rescore the completion under a different
    sampling distribution. It is a symmetric companion to the one-sided
    ``score_completions_under_model`` metric.

    The implementation is deliberately explicit in log-space (it does NOT use
    ``torch.nn.functional.kl_div``, which has historically confusing argument order)
    and performs the divergence reduction in float32 for numerical stability.

    Args:
        base_model: HF CausalLM loaded on a single device, in ``.eval()`` mode.
        base_tokenizer: Tokenizer for the base model. Used for pre-tokenization.
        eval_model: HF CausalLM loaded on the SAME device as ``base_model``.
        eval_tokenizer: Tokenizer for the eval (FT) model. Must be vocab-compatible
            with ``base_tokenizer`` (this is asserted).
        prompts: Already chat-templated prompt strings (the same strings vLLM saw
            during generation). Tokenized with ``add_special_tokens=False``.
        completions: Raw generated text from vLLM, one per prompt.
        batch_size: Mini-batch size. Defaults to 1 because JSD mode holds TWO models
            simultaneously resident on the same GPU.
        max_length: Truncation budget for prompt+completion. If exceeded, the PROMPT
            is front-truncated; the completion is never truncated.
        jsd_log_clamp: Floor used when taking ``log(m)`` to prevent ``-inf`` from
            fp16 underflow. The theoretical mixture ``m`` is strictly positive.

    Returns:
        A list of per-example dicts, in input order::

            {"jsd_sum": float, "jsd_mean": float, "n_tokens": int}

        Empty completions yield ``n_tokens=0``, ``jsd_sum=0.0``,
        ``jsd_mean=float('nan')``. ``jsd_mean`` for non-empty entries is in
        ``[0, log 2] ≈ [0, 0.693]`` (natural log).
    """
    import torch
    import torch.nn.functional as F

    assert len(prompts) == len(completions), (
        f"prompts and completions must have the same length, got {len(prompts)} vs {len(completions)}"
    )

    # Defensive tokenizer compatibility check. Token-level JSD requires the two
    # models to share a vocabulary; otherwise the softmax outputs are not aligned.
    _assert_tokenizers_compatible_for_jsd(
        base_tokenizer,
        eval_tokenizer,
        sample_strings=list(prompts[:4]) + list(completions[:4]),
    )

    pad_id = base_tokenizer.pad_token_id
    if pad_id is None:
        pad_id = base_tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError(
            "base tokenizer has neither pad_token_id nor eos_token_id; cannot pad "
            "batches for JSD scoring."
        )

    base_device = next(base_model.parameters()).device
    eval_device = next(eval_model.parameters()).device
    if base_device != eval_device:
        raise ValueError(
            "score_completions_jsd_between_models: base and eval model must be on "
            f"the same device (got base={base_device}, eval={eval_device}); "
            "single-device scoring only in this helper."
        )
    device = base_device

    base_dtype = next(base_model.parameters()).dtype
    eval_dtype = next(eval_model.parameters()).dtype
    use_base_autocast = (device.type == "cuda" and base_dtype in (torch.bfloat16, torch.float16))
    use_eval_autocast = (device.type == "cuda" and eval_dtype in (torch.bfloat16, torch.float16))

    # Tokenize with the base tokenizer (equivalent to eval tokenizer per the
    # compatibility assertion above).
    pre_tokenized = _pretokenize_prompts_and_completions(
        prompts, completions, base_tokenizer, max_length
    )

    results: list[dict] = [None] * len(prompts)  # type: ignore[list-item]

    for batch_start in range(0, len(pre_tokenized), batch_size):
        batch = pre_tokenized[batch_start : batch_start + batch_size]

        nonempty_indices: list[int] = []
        nonempty_inputs: list[list[int]] = []
        nonempty_completion_lengths: list[int] = []
        for j, (p_ids, c_ids) in enumerate(batch):
            global_idx = batch_start + j
            if len(c_ids) == 0:
                results[global_idx] = {
                    "jsd_sum": 0.0,
                    "jsd_mean": float("nan"),
                    "n_tokens": 0,
                }
                continue
            nonempty_indices.append(j)
            nonempty_inputs.append(p_ids + c_ids)
            nonempty_completion_lengths.append(len(c_ids))

        if not nonempty_inputs:
            continue

        max_len = max(len(seq) for seq in nonempty_inputs)
        b = len(nonempty_inputs)
        input_ids = torch.full((b, max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((b, max_len), dtype=torch.long)
        completion_mask = torch.zeros((b, max_len), dtype=torch.long)
        for k, seq in enumerate(nonempty_inputs):
            seq_len = len(seq)
            input_ids[k, :seq_len] = torch.tensor(seq, dtype=torch.long)
            attention_mask[k, :seq_len] = 1
            c_len = nonempty_completion_lengths[k]
            completion_mask[k, seq_len - c_len : seq_len] = 1

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        completion_mask = completion_mask.to(device)

        with torch.no_grad():
            # Two separate forwards — the models are independent objects.
            if use_base_autocast:
                with torch.autocast(device_type="cuda", dtype=base_dtype):
                    base_out = base_model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                base_out = base_model(input_ids=input_ids, attention_mask=attention_mask)
            base_logits = base_out.logits  # (b, L, V)

            if use_eval_autocast:
                with torch.autocast(device_type="cuda", dtype=eval_dtype):
                    eval_out = eval_model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                eval_out = eval_model(input_ids=input_ids, attention_mask=attention_mask)
            eval_logits = eval_out.logits  # (b, L, V)

            # Standard causal-LM shift: logits at position t predict token t+1.
            # Cast to fp32 for log-softmax + divergence for numerical stability.
            shift_base_logits = base_logits[:, :-1, :].float()
            shift_eval_logits = eval_logits[:, :-1, :].float()
            # Free the original bf16 logits tensors; Python refcount will drop them.
            del base_out, eval_out, base_logits, eval_logits

            shift_completion_mask = completion_mask[:, 1:].float()
            shift_attention_mask = attention_mask[:, 1:].float()
            mask = shift_completion_mask * shift_attention_mask  # (b, L-1)
            mask_bool = mask.bool()                                # (b, L-1)

            # Gather only the positions we actually care about to avoid allocating
            # full-vocab p/q/m/log_m tensors over the entire (b, L-1) grid.
            active_base_logits = shift_base_logits[mask_bool]  # (N_active, V)
            active_eval_logits = shift_eval_logits[mask_bool]  # (N_active, V)
            del shift_base_logits, shift_eval_logits

            if active_base_logits.numel() == 0:
                # No scorable positions across this entire mini-batch. Record
                # per-row empties and move on.
                per_example_n_cpu = mask.sum(dim=-1).long().detach().cpu().tolist()
                for k, j in enumerate(nonempty_indices):
                    global_idx = batch_start + j
                    results[global_idx] = {
                        "jsd_sum": 0.0,
                        "jsd_mean": float("nan"),
                        "n_tokens": int(per_example_n_cpu[k]),
                    }
                continue

            # Explicit symmetric JSD in fp32.
            log_p = F.log_softmax(active_base_logits, dim=-1)  # (N_active, V)
            log_q = F.log_softmax(active_eval_logits, dim=-1)  # (N_active, V)
            del active_base_logits, active_eval_logits
            p = log_p.exp()
            q = log_q.exp()
            m = 0.5 * (p + q)
            log_m = m.clamp_min(jsd_log_clamp).log()
            kl_pm = (p * (log_p - log_m)).sum(dim=-1)  # (N_active,)
            kl_qm = (q * (log_q - log_m)).sum(dim=-1)  # (N_active,)
            del p, q, m, log_m, log_p, log_q
            active_jsd = 0.5 * (kl_pm + kl_qm)          # (N_active,)
            del kl_pm, kl_qm

            # Per-row token counts on the masked grid.
            per_example_n = mask.sum(dim=-1).long()     # (b,)

        active_jsd_cpu = active_jsd.detach().cpu().tolist()
        per_example_n_cpu = per_example_n.detach().cpu().tolist()

        cursor = 0
        for k, j in enumerate(nonempty_indices):
            global_idx = batch_start + j
            n_k = int(per_example_n_cpu[k])
            if n_k <= 0:
                results[global_idx] = {
                    "jsd_sum": 0.0,
                    "jsd_mean": float("nan"),
                    "n_tokens": 0,
                }
                continue
            row_slice = active_jsd_cpu[cursor : cursor + n_k]
            cursor += n_k
            jsd_sum = float(sum(row_slice))
            jsd_mean = jsd_sum / n_k
            results[global_idx] = {
                "jsd_sum": jsd_sum,
                "jsd_mean": jsd_mean,
                "n_tokens": n_k,
            }
        # Defensive: cursor should have consumed all active JSD values this batch.
        assert cursor == len(active_jsd_cpu), (
            f"score_completions_jsd_between_models: cursor mismatch "
            f"({cursor} != {len(active_jsd_cpu)})"
        )

    assert all(r is not None for r in results), (
        "score_completions_jsd_between_models: missing results"
    )
    return results  # type: ignore[return-value]


def _self_check_jsd(small_model_path: str = "Qwen/Qwen2.5-0.5B-Instruct") -> None:
    """Lightweight standalone smoke test for ``score_completions_jsd_between_models``.

    Loads the same small HF model twice on the same device (two independent
    objects with identical weights) and asserts:
      - non-empty records produce near-zero JSD (identical distributions),
      - empty completions produce ``n_tokens=0`` and ``jsd_mean=nan``,
      - output record count matches input count.

    Not a pytest test; invoke manually via::

        python -c "from src.utils.eval_utils import _self_check_jsd; _self_check_jsd()"
    """
    import math
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"[jsd self-check] loading '{small_model_path}' twice on device={device}, dtype={dtype}")
    base_tok = AutoTokenizer.from_pretrained(small_model_path, trust_remote_code=True)
    eval_tok = AutoTokenizer.from_pretrained(small_model_path, trust_remote_code=True)
    base_model = (
        AutoModelForCausalLM.from_pretrained(small_model_path, torch_dtype=dtype, trust_remote_code=True)
        .to(device)
        .eval()
    )
    eval_model = (
        AutoModelForCausalLM.from_pretrained(small_model_path, torch_dtype=dtype, trust_remote_code=True)
        .to(device)
        .eval()
    )

    prompts = [
        "The capital of France is",
        "2 + 2 =",
        "Once upon a time",
    ]
    completions = [
        " Paris, a beautiful city known for the Eiffel Tower.",
        " 4.",
        "",  # empty completion on purpose
    ]

    records = score_completions_jsd_between_models(
        base_model,
        base_tok,
        eval_model,
        eval_tok,
        prompts,
        completions,
        batch_size=1,
        max_length=256,
    )

    assert len(records) == len(prompts), (
        f"[jsd self-check] record count mismatch: {len(records)} != {len(prompts)}"
    )
    # Record for the empty completion.
    empty_rec = records[2]
    assert empty_rec["n_tokens"] == 0, f"[jsd self-check] expected n_tokens=0, got {empty_rec}"
    assert math.isnan(empty_rec["jsd_mean"]), (
        f"[jsd self-check] expected jsd_mean=nan for empty, got {empty_rec}"
    )
    assert empty_rec["jsd_sum"] == 0.0, f"[jsd self-check] expected jsd_sum=0, got {empty_rec}"

    # Non-empty records: identical models => JSD ~ 0.
    for idx in (0, 1):
        rec = records[idx]
        assert rec["n_tokens"] > 0, f"[jsd self-check] expected n_tokens>0, got {rec}"
        assert abs(rec["jsd_mean"]) < 1e-4, (
            f"[jsd self-check] identical-model JSD not near zero at idx={idx}: {rec['jsd_mean']} "
            f"(tolerance 1e-4 nats)"
        )

    # Free models eagerly so the smoke test can be run repeatedly in one process.
    del base_model, eval_model, base_tok, eval_tok
    import gc as _gc
    _gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("[jsd self-check] OK")


def read_jsonl(path: str) -> tuple[list[dict], dict | None]:
    """Read a JSONL eval-results file. Returns (records, summary).

    The summary is the record carrying the ``__summary__: True`` sentinel
    (None if absent). Empty lines are skipped.
    """
    records: list[dict] = []
    summary: dict | None = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("__summary__"):
                summary = obj
            else:
                records.append(obj)
    return records, summary


def load_results_dir(directory: str, *, strict: bool = True) -> tuple[list[dict], dict | None, str]:
    """Load the unique ``*.jsonl`` eval-results file from ``directory``.

    Returns ``(records, summary, filepath)``.

    When ``strict=True`` (default), prints to stderr and ``sys.exit(1)`` if
    zero or multiple JSONL files are found. When ``strict=False``, returns
    ``([], None, "")`` silently if no JSONL is found, but still errors on >1.
    """
    jsonl_files = glob.glob(f"{directory}/*.jsonl")
    if not jsonl_files:
        if strict:
            print(f"ERROR: No .jsonl files found in {directory}", file=sys.stderr)
            sys.exit(1)
        return [], None, ""
    if len(jsonl_files) > 1:
        print(f"ERROR: Multiple .jsonl files in {directory}: {jsonl_files}", file=sys.stderr)
        sys.exit(1)
    records, summary = read_jsonl(jsonl_files[0])
    return records, summary, jsonl_files[0]


# --------------------------------------------------------------------------- #
# Code-generation evaluation harness (extraction, sandboxed exec, test build)  #
# --------------------------------------------------------------------------- #


def extract_code(text: str) -> str:
    """Extract code from model output.

    Priority:
    1. JSON {"code": "..."} block
    2. ```python fenced block
    3. ``` fenced block (any language)
    4. Raw text stripped
    """
    try:
        parsed = json.loads(text.strip())
        if isinstance(parsed, dict) and "code" in parsed:
            return parsed["code"]
    except (json.JSONDecodeError, ValueError):
        pass

    match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    match = re.search(r"```\w*\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    return text.strip()


class TimeoutException(Exception):
    pass


@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def run_code_test(code: str, test_code: str, timeout_s: int = 5) -> tuple[bool, str]:
    """Execute code + test_code in-process with signal-based timeout."""
    try:
        exec_globals = {}
        with time_limit(timeout_s):
            exec(compile(code + "\n" + test_code, "<eval>", "exec"), exec_globals)
        return True, ""
    except TimeoutException:
        return False, f"Timeout after {timeout_s}s"
    except Exception as e:
        return False, str(e)[:500]


def build_mbpp_test_code(problem: dict) -> str:
    """Build test code string for an MBPP problem.

    Composes: test_imports_or_setup + test_list assertions.
    The generated/reference code is prepended separately by the caller.
    """
    setup = problem.get("test_imports_or_setup", "")
    if isinstance(setup, list):
        setup = "\n".join(setup)

    test_assertions = "\n".join(problem["test_list"])

    parts = []
    if setup.strip():
        parts.append(setup.strip())
    parts.append(test_assertions)

    return "\n".join(parts)


def build_humaneval_test_code(problem: dict) -> str:
    """Build test code string for a HumanEval problem.

    HumanEval test field contains def check(candidate) with assertions
    AND the call check(entry_point) at the end.

    Detects whether check() is already called; if not, appends it.
    """
    test = problem["test"]
    entry_point = problem["entry_point"]

    if f"check({entry_point})" in test:
        return test

    return test + f"\n\ncheck({entry_point})\n"


def compute_pass_at_1(passed_list: list[bool]) -> float:
    """Simple mean of pass/fail."""
    if not passed_list:
        return 0.0
    return sum(passed_list) / len(passed_list)


if __name__ == "__main__":
    # Running `python -m src.utils.eval_utils` triggers the JSD smoke test.
    _self_check_jsd()
