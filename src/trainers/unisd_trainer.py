"""
UniSDTrainer: supports multiple strategies for self-distillation. 

Modes:

  - AGREEMENT_SEQ_RANDOM / AGREEMENT_SEQ_RETRIEVAL / AGREEMENT_SEQ_INDUCTION: K pre-built fewshot prompts

"""
import logging

import torch
from torch.nn.functional import kl_div

from src.const import *
from src.trainers.base_trainer import SelfDistillationTrainer
from trl.data_utils import maybe_apply_chat_template
from trl.trainer.utils import pad

logger = logging.getLogger(__name__)


class UniSDTrainer(SelfDistillationTrainer):
    """."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Mode comes directly from config
        self.mode = getattr(self.args, "mode", None)
        assert self.mode is not None, "mode must be specified"
        # Always use ref_model as single teacher
        self.teacher_models = [self.ref_model]

        # Runtime caches for drop_demo augmentation
        self._drop_demo_coverage_checked = False
        self._drop_demo_use_fallback = False

        logger.info(
            "UniSDTrainer: mode=%s, gamma=%.2f",
            self.mode, getattr(self.args, "gamma_agreement", 0.0),
        )


    @property
    def _num_agreement_sources(self) -> int:
        """Number of agreement sources (K) for fewshot/context modes."""
        if self.mode in AGREEMENT_MODES:
            return self._num_fewshot_columns
        return 1

    @property
    def _num_fewshot_columns(self) -> int:
        """Count how many fewshot_teacher_prompt_N columns exist in the dataset."""
        if not hasattr(self, '_cached_num_fewshot'):
            cols = self.train_dataset.column_names if self.train_dataset else []
            k = 0
            while f"fewshot_teacher_prompt_{k}" in cols:
                k += 1
            self._cached_num_fewshot = k
        return self._cached_num_fewshot

    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        if self.mode in AGREEMENT_MODES:
            K = self._num_fewshot_columns
            for k in range(K):
                col = f"fewshot_teacher_prompt_{k}"
                if col not in self._signature_columns:
                    self._signature_columns.append(col)

    def _generate_and_score_completions(self, inputs):
        output = super()._generate_and_score_completions(inputs)

        device = self.accelerator.device

        # Tokenize fewshot teacher prompts (if present)
        if self.mode in AGREEMENT_MODES:
            K = self._num_fewshot_columns

            for k in range(K):
                key = f"fewshot_teacher_prompt_{k}"
                if key not in inputs[0]:
                    break
                prompts_k = [x[key] for x in inputs]
                texts_k = [
                    maybe_apply_chat_template({"prompt": p}, self.processing_class)["prompt"]
                    for p in prompts_k
                ]
                tok_out = self.processing_class(
                    text=texts_k, return_tensors="pt", padding=True, padding_side="left",
                    max_length=self.max_prompt_length, truncation=True, add_special_tokens=False,
                )
                tok_out = super(SelfDistillationTrainer, self)._prepare_inputs(tok_out)
                ids_k, mask_k = tok_out["input_ids"], tok_out["attention_mask"]
                ids_list = [p[m].tolist() for p, m in zip(ids_k, mask_k.bool())]
                ids_k = [torch.tensor(ids, device=device) for ids in ids_list]
                mask_k = [torch.ones_like(ids, dtype=torch.long) for ids in ids_k]
                ids_k = pad(ids_k, padding_value=self.pad_token_id, padding_side="left")
                mask_k = pad(mask_k, padding_value=0, padding_side="left")

                output[f"fewshot_teacher_prompt_ids_{k}"] = ids_k
                output[f"fewshot_teacher_prompt_mask_{k}"] = mask_k

        # Tokenize bad_teacher_prompt if present (contrastive mode)
        if "bad_teacher_prompt" in inputs[0] and inputs[0]["bad_teacher_prompt"] is not None:
            bad_teacher_prompts = [x["bad_teacher_prompt"] for x in inputs]
            bad_texts = [
                maybe_apply_chat_template({"prompt": p}, self.processing_class)["prompt"]
                for p in bad_teacher_prompts
            ]
            tok_out = self.processing_class(
                text=bad_texts, return_tensors="pt", padding=True, padding_side="left",
                max_length=self.max_prompt_length, truncation=True, add_special_tokens=False,
            )
            tok_out = super(SelfDistillationTrainer, self)._prepare_inputs(tok_out)
            ids, mask = tok_out["input_ids"], tok_out["attention_mask"]
            ids_list = [p[m].tolist() for p, m in zip(ids, mask.bool())]
            ids = [torch.tensor(t, device=device) for t in ids_list]
            mask = [torch.ones_like(t, dtype=torch.long) for t in ids]
            ids = pad(ids, padding_value=self.pad_token_id, padding_side="left")
            mask = pad(mask, padding_value=0, padding_side="left")
            output["bad_teacher_prompt_ids"] = ids
            output["bad_teacher_prompt_mask"] = mask

        return output

    # ------------------------------------------------------------------
    # Agreement weight computation
    # ------------------------------------------------------------------

    def _compute_agreement_weights(self, teacher_input_ids, teacher_attention_mask,
                                   logits_to_keep, inputs,
                                   teacher_per_token_logps=None, loss_mask=None):
        """Compute per-token agreement weights from multi-teacher logps.

        Returns:
            agreement_weights: (B, T) tensor of weights in [0, 1], or None.
            disagreement: (B, T) tensor of disagreement values, or None.
        """
        if self.args.gamma_agreement <= 0.0:
            return None, None

        if self.mode in AGREEMENT_MODES:
            teacher_logps_list = self._collect_context_logps(
                inputs, logits_to_keep, teacher_per_token_logps
            )
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        stacked = torch.stack(teacher_logps_list, dim=0)  # (K, B, T)

        if self.mode in TOKEN_GRANULARITY_MODES:
            granularity = "token"
        else:
            granularity = "sequence"

        if granularity == "sequence":
            return self._agreement_weights_sequence(stacked, inputs, loss_mask)
        else:
            return self._agreement_weights_token(stacked, inputs, loss_mask)

    # ------------------------------------------------------------------
    # Token-level agreement weights
    # ------------------------------------------------------------------

    def _agreement_weights_token(self, stacked, inputs, loss_mask):
        """Compute per-token agreement weights from stacked teacher logps (K, B, T)."""
        if self.args.agreement_stat == "var":
            disagreement = stacked.var(dim=0)  # (B, T)
        else:  # "range"
            disagreement = stacked.max(dim=0).values - stacked.min(dim=0).values

        mask = loss_mask.bool() if loss_mask is not None else inputs["completion_mask"].bool()
        flat = disagreement[mask].float()

        if flat.numel() == 0:
            agreement_weights = torch.ones_like(disagreement)
        else:
            q = getattr(self.args, "agreement_quantile", 0.95)
            scale = torch.quantile(flat, q).clamp(min=1e-6)

            disagreement_norm = disagreement / scale

            cap = float(getattr(self.args, "agreement_cap", 10.0))
            disagreement_norm = disagreement_norm.clamp(max=cap)

            agreement_weights = 1.0 / (1.0 + float(self.args.gamma_agreement) * disagreement_norm)

        agreement_weights = agreement_weights.clamp(min=0.01)
        return agreement_weights, disagreement

    # ------------------------------------------------------------------
    # Sequence-level agreement weights
    # ------------------------------------------------------------------

    def _agreement_weights_sequence(self, stacked, inputs, loss_mask):
        """Compute per-sequence agreement weights from stacked teacher logps (K, B, T)."""
        mask = loss_mask.bool() if loss_mask is not None else inputs["completion_mask"].bool()

        token_counts = mask.sum(-1).unsqueeze(0).float().clamp(min=1.0)
        seq_score = (stacked * mask.unsqueeze(0)).sum(-1) / token_counts

        if self.args.agreement_stat == "var":
            disagreement_seq = seq_score.var(dim=0, unbiased=False)
        else:
            disagreement_seq = seq_score.max(dim=0).values - seq_score.min(dim=0).values

        w_seq = torch.exp(-self.args.gamma_agreement * disagreement_seq).clamp(min=1e-6)

        T = stacked.size(2)
        agreement_weights = w_seq.unsqueeze(-1).expand(-1, T) * mask.to(w_seq.dtype)

        return agreement_weights, disagreement_seq

    # ------------------------------------------------------------------
    # Batched forward pass helper
    # ------------------------------------------------------------------

    def _batch_forward_logps(self, all_input_ids, all_attention_masks, logits_to_keep, inputs):
        """Run a single batched forward pass for multiple context sequences."""
        B = all_input_ids[0].size(0)
        K = len(all_input_ids)

        max_len = max(ids.size(1) for ids in all_input_ids)
        device = all_input_ids[0].device
        dtype = all_input_ids[0].dtype

        batched_ids = torch.full((K * B, max_len), self.pad_token_id, device=device, dtype=dtype)
        batched_mask = torch.zeros((K * B, max_len), device=device, dtype=torch.long)

        for k in range(K):
            seq_len = all_input_ids[k].size(1)
            offset = max_len - seq_len
            batched_ids[k * B:(k + 1) * B, offset:] = all_input_ids[k]
            batched_mask[k * B:(k + 1) * B, offset:] = all_attention_masks[k]

        with torch.no_grad():
            logps, _, _, _ = self._get_per_token_logps_and_entropies(
                self.ref_model,
                batched_ids,
                batched_mask,
                logits_to_keep,
                compute_entropy=False,
                compute_all_logps=False,
                pixel_values=inputs.get("pixel_values"),
                image_grid_thw=inputs.get("image_grid_thw"),
                num_images=inputs.get("num_images"),
                pixel_attention_mask=inputs.get("pixel_attention_mask"),
                image_sizes=inputs.get("image_sizes"),
                token_type_ids=None,
                logit_temperature=1.0,
            )

        return list(logps.reshape(K, B, -1))

    # ------------------------------------------------------------------
    # Multi-teacher logps collection (batched)
    # ------------------------------------------------------------------

    def _collect_context_logps(self, inputs, logits_to_keep, teacher_per_token_logps):
        """Collect per-token logps from the same teacher under K different contexts.

        All contexts are forward-passed with logit_temperature=1.0 for consistent
        agreement scoring (independent of generation temperature). For fewshot
        modes (UNISD_STAR and AGREEMENT_*), K contexts use pre-built fewshot
        prompts and are batched together at temperature 1.0.
        """
        completion_ids = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]

        has_pixel_data = inputs.get("pixel_values") is not None

        if self.mode not in AGREEMENT_MODES:
            raise ValueError(f"Unknown mode: {self.mode}")

        K = self._num_fewshot_columns
        all_input_ids = []
        all_attention_masks = []
        for ctx_k in range(K):
            aug_prompt_ids = inputs[f"fewshot_teacher_prompt_ids_{ctx_k}"]
            aug_prompt_mask = inputs[f"fewshot_teacher_prompt_mask_{ctx_k}"]
            all_input_ids.append(torch.cat([aug_prompt_ids, completion_ids], dim=1))
            all_attention_masks.append(torch.cat([aug_prompt_mask, completion_mask], dim=1))

        if has_pixel_data:
            teacher_logps_list = self._sequential_forward_logps(
                all_input_ids, all_attention_masks, logits_to_keep, inputs
            )
        else:
            teacher_logps_list = self._batch_forward_logps(
                all_input_ids, all_attention_masks, logits_to_keep, inputs
            )

        self._log_context_diffs(teacher_logps_list, prefix="fewshot context")
        return teacher_logps_list

    def _sequential_forward_logps(self, all_input_ids, all_attention_masks, logits_to_keep, inputs):
        """Fallback: run one forward pass per context sequentially (for VLM models)."""
        results = []
        for ids, mask in zip(all_input_ids, all_attention_masks):
            with torch.no_grad():
                t_logps, _, _, _ = self._get_per_token_logps_and_entropies(
                    self.ref_model, ids, mask, logits_to_keep,
                    compute_entropy=False, compute_all_logps=False,
                    pixel_values=inputs.get("pixel_values"),
                    image_grid_thw=inputs.get("image_grid_thw"),
                    num_images=inputs.get("num_images"),
                    pixel_attention_mask=inputs.get("pixel_attention_mask"),
                    image_sizes=inputs.get("image_sizes"),
                    token_type_ids=None,
                    logit_temperature=1.0,
                )
            results.append(t_logps)
        return results

    def _log_context_diffs(self, teacher_logps_list, prefix="context"):
        """Log pairwise differences between context logps for debugging."""
        if len(teacher_logps_list) < 2:
            return
        with torch.no_grad():
            mode = "train" if self.model.training else "eval"
            others = torch.stack(teacher_logps_list[1:], dim=0)
            abs_diffs = (others - teacher_logps_list[0].unsqueeze(0)).abs()
            per_ctx_max = abs_diffs.amax(dim=(1, 2))
            per_ctx_mean = abs_diffs.mean(dim=(1, 2))
            for ctx_idx in range(per_ctx_max.size(0)):
                logger.debug(
                    "%s_%d vs %s_0: max_abs_diff=%.6f, mean_abs_diff=%.6f",
                    prefix, ctx_idx + 1, prefix,
                    per_ctx_max[ctx_idx].item(), per_ctx_mean[ctx_idx].item(),
                )
            self._metrics[mode]["agreement/context_max_abs_diff"].append(per_ctx_max.max().item())

    # ------------------------------------------------------------------
    # Context augmentation dispatcher and strategies
    # ------------------------------------------------------------------

    def _augment_teacher_context(self, teacher_prompt_ids, teacher_prompt_mask, context_index):
        """Dispatch to the appropriate augmentation strategy."""
        augment = getattr(self.args, "context_augment", "none")

        if augment == "none" or context_index == 0:
            return teacher_prompt_ids.clone(), teacher_prompt_mask.clone()
        elif augment == "token_drop":
            return self._augment_token_drop(
                teacher_prompt_ids, teacher_prompt_mask, context_index
            )
        elif augment == "drop_demo":
            return self._augment_drop_demo(
                teacher_prompt_ids, teacher_prompt_mask, context_index
            )
        else:
            logger.warning(
                "Unknown context_augment='%s', returning original prompt.", augment
            )
            return teacher_prompt_ids.clone(), teacher_prompt_mask.clone()

    def _augment_token_drop(self, teacher_prompt_ids, teacher_prompt_mask, context_index):
        """Drop a contiguous span of tokens from each teacher prompt."""
        batch_size = teacher_prompt_ids.size(0)
        drop_prob = getattr(self.args, "context_drop_prob", 0.5)
        device = teacher_prompt_ids.device
        dtype = teacher_prompt_ids.dtype

        global_step = getattr(self, "state", None)
        global_step = global_step.global_step if global_step is not None else 0
        base_seed = getattr(self.args, "seed", 42) or 42

        new_ids_list = []
        new_mask_list = []
        total_dropped = 0

        for i in range(batch_size):
            mask_i = teacher_prompt_mask[i].bool()
            ids_i = teacher_prompt_ids[i]

            valid_ids = ids_i[mask_i]
            num_valid = valid_ids.size(0)

            min_keep = 8
            if num_valid <= min_keep:
                new_ids_list.append(valid_ids)
                new_mask_list.append(torch.ones(num_valid, device=device, dtype=torch.long))
                continue

            rng = torch.Generator(device="cpu")
            rng.manual_seed(base_seed + context_index * 10000 + i + global_step * 100)

            max_drop = max(1, num_valid - min_keep)
            drop_length = max(1, min(int(num_valid * drop_prob), max_drop))

            max_start = num_valid - drop_length
            start_pos = int(torch.randint(0, max_start + 1, (1,), generator=rng).item())

            kept = torch.cat([valid_ids[:start_pos], valid_ids[start_pos + drop_length:]])
            new_ids_list.append(kept)
            new_mask_list.append(torch.ones(kept.size(0), device=device, dtype=torch.long))
            total_dropped += drop_length

        max_len = max(t.size(0) for t in new_ids_list)
        padded_ids = torch.full((batch_size, max_len), self.pad_token_id, device=device, dtype=dtype)
        padded_mask = torch.zeros((batch_size, max_len), device=device, dtype=torch.long)

        for i in range(batch_size):
            seq_len = new_ids_list[i].size(0)
            padded_ids[i, max_len - seq_len:] = new_ids_list[i]
            padded_mask[i, max_len - seq_len:] = new_mask_list[i]

        avg_dropped = total_dropped / batch_size
        logger.debug(
            "token_drop ctx=%d: avg_dropped=%.1f tokens, new_len=%d",
            context_index, avg_dropped, max_len,
        )

        return padded_ids, padded_mask

    def _augment_drop_demo(self, teacher_prompt_ids, teacher_prompt_mask, context_index):
        """Remove the demonstration section from teacher prompts."""
        batch_size = teacher_prompt_ids.size(0)
        device = teacher_prompt_ids.device
        dtype = teacher_prompt_ids.dtype

        start_marker = "This is an example for a response to the question:\n"
        end_marker = f"\n{SELF_DISTILLATION_INSTRUCTION}"

        if not self._drop_demo_coverage_checked:
            self._drop_demo_coverage_checked = True
            check_count = min(32, batch_size)
            marker_found = 0
            for i in range(check_count):
                mask_i = teacher_prompt_mask[i].bool()
                ids_i = teacher_prompt_ids[i][mask_i]
                text = self.processing_class.decode(ids_i, skip_special_tokens=False)
                if start_marker in text and end_marker in text:
                    marker_found += 1
            coverage = marker_found / check_count if check_count > 0 else 0.0
            logger.info(
                "drop_demo coverage diagnostic: marker_coverage_ratio=%.2f (%d/%d)",
                coverage, marker_found, check_count,
            )
            if coverage < 0.5:
                logger.warning(
                    "Low marker coverage (%.2f). Falling back to newline-split for drop_demo.",
                    coverage,
                )
                self._drop_demo_use_fallback = True
            else:
                self._drop_demo_use_fallback = False

        max_prompt_length = getattr(self.args, "max_prompt_length", 1024)
        new_ids_list = []
        new_mask_list = []
        augmented_count = 0

        rng = torch.Generator(device="cpu")
        base_seed = getattr(self.args, "seed", 42) or 42
        global_step = getattr(self, "state", None)
        global_step = global_step.global_step if global_step is not None else 0

        for i in range(batch_size):
            mask_i = teacher_prompt_mask[i].bool()
            ids_i = teacher_prompt_ids[i][mask_i]
            text = self.processing_class.decode(ids_i, skip_special_tokens=False)
            new_text = text
            applied = False

            if not self._drop_demo_use_fallback:
                s_idx = text.find(start_marker)
                e_idx = text.find(end_marker)
                if s_idx != -1 and e_idx != -1 and e_idx > s_idx:
                    new_text = text[:s_idx] + text[e_idx:]
                    applied = True

            if not applied:
                blocks = text.split("\n\n")
                if len(blocks) > 2:
                    rng.manual_seed(base_seed + context_index * 10000 + i + global_step * 100)
                    drop_idx = int(torch.randint(0, len(blocks), (1,), generator=rng).item())
                    new_blocks = [b for j, b in enumerate(blocks) if j != drop_idx]
                    new_text = "\n\n".join(new_blocks)
                    applied = True

            if applied:
                augmented_count += 1

            encoded = self.processing_class(
                new_text,
                truncation=True,
                max_length=max_prompt_length,
                add_special_tokens=False,
                return_tensors="pt",
            )
            tok_ids = encoded["input_ids"].squeeze(0).to(device=device, dtype=dtype)
            new_ids_list.append(tok_ids)
            new_mask_list.append(torch.ones(tok_ids.size(0), device=device, dtype=torch.long))

        max_len = max(t.size(0) for t in new_ids_list)
        padded_ids = torch.full((batch_size, max_len), self.pad_token_id, device=device, dtype=dtype)
        padded_mask = torch.zeros((batch_size, max_len), device=device, dtype=torch.long)

        for i in range(batch_size):
            seq_len = new_ids_list[i].size(0)
            padded_ids[i, max_len - seq_len:] = new_ids_list[i]
            padded_mask[i, max_len - seq_len:] = new_mask_list[i]

        aug_ratio = augmented_count / batch_size if batch_size > 0 else 0.0
        logger.debug(
            "drop_demo ctx=%d: augmentation_applied_ratio=%.2f, new_len=%d",
            context_index, aug_ratio, max_len,
        )

        mode = "train" if self.model.training else "eval"
        self._metrics[mode]["agreement/augmentation_applied_ratio"].append(aug_ratio)

        return padded_ids, padded_mask

    # ------------------------------------------------------------------
    # Loss override (surgical copy of parent + agreement injection)
    # ------------------------------------------------------------------

    def _compute_loss(self, model, inputs):
        # ---- Verbatim from parent: unpack inputs ----
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        teacher_prompt_ids, teacher_prompt_mask = inputs["teacher_prompt_ids"], inputs["teacher_prompt_mask"]

        # Create a separate mask for loss computation that skips the first N tokens
        loss_completion_mask = completion_mask
        if self.num_loss_tokens_to_skip > 0:
            batch_size, seq_len = completion_mask.shape
            token_positions = torch.arange(seq_len, device=completion_mask.device).unsqueeze(0).expand(batch_size, -1)
            skip_mask = (token_positions >= self.num_loss_tokens_to_skip).int()
            loss_completion_mask = completion_mask * skip_mask

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        teacher_input_ids = torch.cat([teacher_prompt_ids, completion_ids], dim=1)
        teacher_attention_mask = torch.cat([teacher_prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        # ---- Verbatim: student forward ----
        # Request hidden states only when final-layer distillation mode is active
        return_hidden_states = self.mode in {MATCH_JOINT, MATCH_REPR, UNISD_STAR}
        per_token_logps, all_logps, entropies, student_hidden = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=True,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            num_images=inputs.get("num_images"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            image_sizes=inputs.get("image_sizes"),
            token_type_ids=inputs.get("token_type_ids"),
            return_hidden_states=return_hidden_states,
        )

        # ---- Verbatim: primary teacher (ref_model) forward ----
        with torch.no_grad():
            teacher_per_token_logps, teacher_all_logps, teacher_entropies, teacher_hidden = self._get_per_token_logps_and_entropies(
                self.ref_model,
                teacher_input_ids,
                teacher_attention_mask,
                logits_to_keep,
                compute_entropy=True,
                pixel_values=inputs.get("pixel_values"),
                image_grid_thw=inputs.get("image_grid_thw"),
                num_images=inputs.get("num_images"),
                pixel_attention_mask=inputs.get("pixel_attention_mask"),
                image_sizes=inputs.get("image_sizes"),
                token_type_ids=inputs.get("token_type_ids"),
                return_hidden_states=return_hidden_states,
            )

        # ---- INJECTED: compute agreement weights ----
        agreement_weights, disagreement = self._compute_agreement_weights(
            teacher_input_ids, teacher_attention_mask, logits_to_keep, inputs,
            teacher_per_token_logps=teacher_per_token_logps, loss_mask=loss_completion_mask,
        )

        # ---- Verbatim: entropy mask ----
        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(entropies, loss_completion_mask, 1 - self.top_entropy_quantile)
        else:
            entropy_mask = None

        # ---- Verbatim: KL to base model (beta regularizer) ----
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        # ---- Verbatim: KL divergence loss ----
        if self.alpha == 0:  # Forward KL
            kl_loss = kl_div(all_logps, teacher_all_logps, reduction="none", log_target=True)
        elif self.alpha == 1:  # Reverse KL
            kl_loss = kl_div(teacher_all_logps, all_logps, reduction="none", log_target=True)
        else:
            alpha = torch.tensor(self.alpha, dtype=all_logps.dtype)
            mixture_log_probs = torch.logsumexp(
                torch.stack([all_logps + torch.log(1 - alpha), teacher_all_logps + torch.log(alpha)]),
                dim=0,
            )
            kl_teacher = kl_div(mixture_log_probs, teacher_all_logps, reduction="none", log_target=True)
            kl_student = kl_div(mixture_log_probs, all_logps, reduction="none", log_target=True)
            kl_loss = alpha * kl_teacher + (1 - alpha) * kl_student

        per_token_loss = kl_loss.sum(-1)

        # Applied right after vocab-reduction, before any masking / weighting / final reduction.
        # Defaults (both None) are no-op → regression-safe.
        token_clip = getattr(self.args, "token_clip", None)
        token_clip_quantile = getattr(self.args, "token_clip_quantile", None)
        if token_clip_quantile is not None:
            mode_for_log = "train" if self.model.training else "eval"
            with torch.no_grad():
                flat = per_token_loss[loss_completion_mask.bool()].float()
                if flat.numel() > 0:
                    cap = torch.quantile(flat, token_clip_quantile).clamp(min=1e-6)
                    self._metrics[mode_for_log]["clip/adaptive_cap"].append(cap.item())
                    cap_val = cap.item()
                else:
                    cap_val = None
            if cap_val is not None:
                per_token_loss = per_token_loss.clamp(max=cap_val)
        elif token_clip is not None:
            per_token_loss = per_token_loss.clamp(max=token_clip)

        # ---- INJECTED: apply agreement weights to per_token_loss ----
        if agreement_weights is not None:
            if entropy_mask is not None:
                agreement_weights = torch.where(entropy_mask, agreement_weights, torch.ones_like(agreement_weights))

            per_token_loss = per_token_loss * agreement_weights

        # ---- Verbatim: importance sampling correction ----
        if self.use_vllm and self.vllm_importance_sampling_correction and not self.generate_from_teacher:
            ratio = inputs["importance_sampling_ratio"]
            importance_weights = (ratio * loss_completion_mask).sum(-1) / loss_completion_mask.sum(-1).clamp(min=1.0)
            importance_weights = importance_weights.unsqueeze(-1)
            per_token_loss = per_token_loss * importance_weights

        # ---- Verbatim: entropy mask ----
        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask

        # ---- Final loss ----
        if agreement_weights is not None:
            weighted_mask = loss_completion_mask * agreement_weights
            loss = ((per_token_loss * loss_completion_mask).sum(-1) / weighted_mask.sum(-1).clamp(min=1.0)).mean()
        else:
            loss = ((per_token_loss * loss_completion_mask).sum(-1) / loss_completion_mask.sum(-1).clamp(min=1.0)).mean()
        loss = loss / self.current_gradient_accumulation_steps

        # ---- Verbatim: standard logging ----
        mode = "train" if self.model.training else "eval"

        with torch.no_grad():
            delta = teacher_per_token_logps - per_token_logps
            delta = delta.clamp(min=-50, max=50)
            kl_approx = (-delta) + torch.exp(delta) - 1

            mask = loss_completion_mask.bool()
            kl_vals = kl_approx[mask]
            kl_vals = kl_vals[torch.isfinite(kl_vals)]
            kl_approx_mean = kl_vals.mean() if kl_vals.numel() else torch.tensor(0.0, device=kl_approx.device)
            assert torch.isfinite(kl_approx_mean), f"kl_approx_mean is NaN/Inf: {kl_approx_mean}"

        self._metrics[mode]["kl_approx"].append(self.accelerator.gather(kl_approx_mean).nanmean().item())

        loss_completion_token_count = loss_completion_mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            if x.shape[1] == 1:
                return x.mean()
            else:
                return (x * loss_completion_mask).sum() / loss_completion_token_count

        if self.beta != 0.0:
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl_to_base_model"].append(self.accelerator.gather(mean_kl).nanmean().item())

        mean_entropy = masked_batch_mean(entropies)
        self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())

        # ---- INJECTED: agreement stats logging ----
        if getattr(self.args, "log_agreement_stats", True):
            self._log_agreement_stats(mode, agreement_weights, disagreement, loss_completion_mask)

        # ---- INJECTED: contrastive loss ----
        lambda_c = getattr(self.args, "contrastive_weight", 0.0)
        if lambda_c > 0.0 and "bad_teacher_prompt_ids" in inputs:
            bad_teacher_prompt_ids = inputs["bad_teacher_prompt_ids"]
            bad_teacher_prompt_mask = inputs["bad_teacher_prompt_mask"]
            bad_teacher_input_ids = torch.cat([bad_teacher_prompt_ids, completion_ids], dim=1)
            bad_teacher_attention_mask = torch.cat([bad_teacher_prompt_mask, completion_mask], dim=1)

            with torch.no_grad():
                bad_teacher_per_token_logps, _, _, _ = self._get_per_token_logps_and_entropies(
                    self.ref_model,
                    bad_teacher_input_ids, bad_teacher_attention_mask,
                    logits_to_keep,
                    compute_entropy=False, compute_all_logps=False,
                    pixel_values=inputs.get("pixel_values"),
                    image_grid_thw=inputs.get("image_grid_thw"),
                    num_images=inputs.get("num_images"),
                    pixel_attention_mask=inputs.get("pixel_attention_mask"),
                    image_sizes=inputs.get("image_sizes"),
                    token_type_ids=None,
                    # logit_temperature=1.0,
                )

            margin = getattr(self.args, "contrastive_margin", 0.5)
            
            pos_dist = (per_token_logps - teacher_per_token_logps).abs()
            neg_dist = (per_token_logps - bad_teacher_per_token_logps).abs()
            per_token_contrastive = torch.relu(margin + pos_dist - neg_dist)

            contrastive_loss = (
                (per_token_contrastive * loss_completion_mask).sum(-1)
                / loss_completion_mask.sum(-1).clamp(min=1.0)
            ).mean() / self.current_gradient_accumulation_steps

            loss = loss + lambda_c * contrastive_loss

            # Logging
            with torch.no_grad():
                mask = loss_completion_mask.bool()
                pos_vals = pos_dist[mask]
                neg_vals = neg_dist[mask]
                ctr_vals = per_token_contrastive[mask]
                bad_vals = bad_teacher_per_token_logps[mask]
                good_vals = teacher_per_token_logps[mask]
                stu_vals = per_token_logps[mask]

                self._metrics[mode]["contrastive/loss"].append(
                    self.accelerator.gather(
                        contrastive_loss * self.current_gradient_accumulation_steps
                    ).nanmean().item()
                )
                self._metrics[mode]["contrastive/pos_dist_mean"].append(
                    self.accelerator.gather(pos_vals.mean()).nanmean().item() if pos_vals.numel() > 0 else 0.0
                )
                self._metrics[mode]["contrastive/neg_dist_mean"].append(
                    self.accelerator.gather(neg_vals.mean()).nanmean().item() if neg_vals.numel() > 0 else 0.0
                )
                self._metrics[mode]["contrastive/margin_violation_mean"].append(
                    self.accelerator.gather(ctr_vals.mean()).nanmean().item() if ctr_vals.numel() > 0 else 0.0
                )
                self._metrics[mode]["contrastive/bad_teacher_logp_mean"].append(
                    self.accelerator.gather(bad_vals.mean()).nanmean().item() if bad_vals.numel() > 0 else 0.0
                )
                self._metrics[mode]["contrastive/good_teacher_logp_mean"].append(
                    self.accelerator.gather(good_vals.mean()).nanmean().item() if good_vals.numel() > 0 else 0.0
                )
                self._metrics[mode]["contrastive/student_logp_mean"].append(
                    self.accelerator.gather(stu_vals.mean()).nanmean().item() if stu_vals.numel() > 0 else 0.0
                )

        # ---- INJECTED: final-layer hidden-state distillation loss ----
        if self.mode in {MATCH_JOINT, MATCH_REPR, UNISD_STAR}:
            assert student_hidden is not None and teacher_hidden is not None, f"For mode {self.mode}, student_hidden and teacher_hidden must be not None"
            # Masked MSE over completion tokens (hidden states already completion-only)
            per_token_hidden_mse = ((student_hidden - teacher_hidden.detach()) ** 2).mean(dim=-1)  # (B, T)
            hidden_loss = (
                (per_token_hidden_mse * loss_completion_mask).sum(-1)
                / loss_completion_mask.sum(-1).clamp(min=1.0)
            ).mean() / self.current_gradient_accumulation_steps
            if self.mode == MATCH_REPR:
                loss = hidden_loss
            else:
                fl_weight = getattr(self.args, "final_layer_distill_weight", 0.1)
                loss = loss + fl_weight * hidden_loss

            # Logging
            with torch.no_grad():
                self._metrics[mode]["final_layer_distill/loss"].append(
                    self.accelerator.gather(
                        hidden_loss * self.current_gradient_accumulation_steps
                    ).nanmean().item()
                )

        return loss

    # ------------------------------------------------------------------
    # Agreement stats logging
    # ------------------------------------------------------------------

    def _log_agreement_stats(self, mode, agreement_weights, disagreement, loss_completion_mask):
        """Log agreement weight and disagreement statistics."""
        with torch.no_grad():
            if agreement_weights is not None and disagreement is not None:
                mask = loss_completion_mask.bool()
                masked_weights = agreement_weights[mask]

                if disagreement.dim() == 1:
                    # Sequence-level: stats over batch dimension
                    if disagreement.numel() > 0:
                        d_mean = disagreement.mean()
                        d_min = disagreement.min()
                        d_max = disagreement.max()
                    else:
                        d_mean = d_min = d_max = torch.tensor(0.0)
                else:
                    # Token-level: mask with loss_completion_mask
                    masked_disagreement = disagreement[mask]
                    if masked_disagreement.numel() > 0:
                        d_mean = masked_disagreement.mean()
                        d_min = masked_disagreement.min()
                        d_max = masked_disagreement.max()
                    else:
                        d_mean = d_min = d_max = torch.tensor(0.0)

                if masked_weights.numel() > 0:
                    w_mean = masked_weights.mean()
                    w_min = masked_weights.min()
                    w_max = masked_weights.max()
                else:
                    w_mean = w_min = w_max = torch.tensor(1.0)

                self._metrics[mode]["agreement/weight_mean"].append(
                    self.accelerator.gather(w_mean).nanmean().item())
                self._metrics[mode]["agreement/weight_min"].append(
                    self.accelerator.gather(w_min).nanmean().item())
                self._metrics[mode]["agreement/weight_max"].append(
                    self.accelerator.gather(w_max).nanmean().item())
                self._metrics[mode]["agreement/disagreement_mean"].append(
                    self.accelerator.gather(d_mean).nanmean().item())
                self._metrics[mode]["agreement/disagreement_min"].append(
                    self.accelerator.gather(d_min).nanmean().item())
                self._metrics[mode]["agreement/disagreement_max"].append(
                    self.accelerator.gather(d_max).nanmean().item())
            else:
                self._metrics[mode]["agreement/weight_mean"].append(1.0)
                self._metrics[mode]["agreement/weight_min"].append(1.0)
                self._metrics[mode]["agreement/weight_max"].append(1.0)
                self._metrics[mode]["agreement/disagreement_mean"].append(0.0)
                self._metrics[mode]["agreement/disagreement_min"].append(0.0)
                self._metrics[mode]["agreement/disagreement_max"].append(0.0)
