"""
UniSD entry point.
Uses --mode to select a UniSD training variant (e.g. unisd_star, ema, contrastive, match_repr,
agreement_seq_random, agreement_seq_retrieval, agreement_seq_induction, clip).
"""
import os
import sys
import time
import warnings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
os.environ["TRL_EXPERIMENTAL_SILENCE"] = "1"
warnings.filterwarnings("ignore", message="incompatible copy of pydevd")
import torch
import json
import random
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.analysis.resource_consumption_utils import (
    append_resource_record,
    compute_resource_consumption_record,
    detect_num_gpus,
)
from src.utils.log_utils import extract_final_train_metrics

from src.const import *
from src.trainers.unisd_trainer import UniSDTrainer
from src.config.multi_teacher_config import UniSDConfig
from src.utils.lora_utils import get_default_lora_config
from src.utils.env_utils import disable_thinking_mode
from src.utils.path_utils import get_induction_cache_path, get_model_dir, build_hparam_tag

from src.utils.tooluse_utils import load_tooluse_dataset
from src.data.mbpp import load_mbpp_training_dataset
from src.data.scienceqa import load_scienceqa_training_dataset
from src.data.cos_e import load_cos_e_training_dataset
from src.data.medmcqa import load_medmcqa_training_dataset

from src.train.train_args import parse_args, apply_mode_defaults


def attach_bad_teacher_prompts(train_dataset, cache_path, dataset_name, seed=42):
    """Load negative demo cache and attach bad_teacher_prompt to each training example.

    For each example, randomly samples ONE of the available negative demos (typically 4).

    For MCQA datasets (gpqa, scienceqa): uses build_teacher_text_gpqa/scienceqa with
    the wrong answer from the sampled negative demo.
    For MBPP: uses build_teacher_text with the buggy code answer.
    """

    with open(cache_path) as f:
        neg_cache = json.load(f)
    print(f"[NegDemo] Loaded {len(neg_cache)} negative demo entries from {cache_path}")

    # Import the appropriate teacher text builder
    if dataset_name in (GPQA, SCIENCEQA, COS_E, MEDMCQA):
        from src.data.scienceqa import build_teacher_text_scienceqa as _build_teacher
    elif dataset_name == MBPP:
        from src.data.mbpp import build_teacher_text as _build_teacher
    elif dataset_name == TOOLUSE:
        from src.utils.tooluse_utils import build_teacher_text_tooluse as _build_teacher
    else:
        raise ValueError(f"Contrastive mode not yet supported for dataset: {dataset_name}")

    rng = random.Random(seed)

    # Convert HF dataset to list of dicts for mutation
    data_list = [dict(train_dataset[i]) for i in range(len(train_dataset))]
    result = []

    for example in data_list:
        task_id = example.get("task_id")
        if task_id is None or str(task_id) not in neg_cache:
            continue

        neg_demos = neg_cache[str(task_id)]
        if not neg_demos:
            continue

        neg = rng.choice(neg_demos)
        student_text = example["prompt"][0]["content"]

        if dataset_name in (GPQA, SCIENCEQA, COS_E, MEDMCQA):
            wrong_letter = neg.get("target_wrong_letter", neg.get("answer", ""))
            wrong_reasoning = neg.get("reasoning", "")
            bad_text = _build_teacher(student_text, wrong_letter, wrong_reasoning)
        elif dataset_name == MBPP:
            buggy_code = neg.get("answer", "")
            bad_text = _build_teacher(student_text, buggy_code)
        elif dataset_name == TOOLUSE:
            wrong_answer = neg.get("answer", "")
            wrong_reasoning = neg.get("reasoning", "")
            bad_text = _build_teacher(student_text, wrong_answer, wrong_reasoning)
        else:
            continue

        example["bad_teacher_prompt"] = [{"role": "user", "content": bad_text}]
        result.append(example)

    dropped = len(data_list) - len(result)
    print(f"[NegDemo] Attached bad_teacher_prompt to {len(result)}/{len(data_list)} examples"
          + (f" (dropped {dropped} without neg demos)" if dropped else ""))

    from datasets import Dataset
    return Dataset.from_list(result)


def main():
    args = parse_args()
    t0 = time.time()
    apply_mode_defaults(args)

    num_auxiliary_contexts = args.num_auxiliary_contexts

    assert not args.dataset == GPQA, "GPQA CANNOT be used for training"

    # Load dataset
    if args.dataset == MBPP:
        # Construct induction cache path if needed
        induction_cache_file = None
        if args.fewshot_method == INDUCTION:
            induction_model = args.induction_model_name or args.model_name
            induction_cache_file = get_induction_cache_path(output_dir=args.output_dir, model_name=induction_model, dataset=args.dataset, num_auxiliary_contexts=num_auxiliary_contexts, num_demos=args.induction_num_demos)
            
            assert os.path.exists(induction_cache_file), f"Induction cache file not found: {induction_cache_file}"
        
        train_dataset, eval_dataset = load_mbpp_training_dataset(
            seed=args.seed,
            max_train_samples=args.num_train_examples,
            max_eval_samples=args.num_eval_examples,
            config=args.mbpp_config,
            num_auxiliary_contexts=num_auxiliary_contexts,
            fewshot_method=args.fewshot_method,
            retrieval_model=args.retrieval_model,
            embedding_cache_dir=args.embedding_cache_dir,
            demo_k=args.demo_k,
            induction_cache_file=induction_cache_file,
        )
        
        
    elif args.dataset == TOOLUSE:
        # Construct induction cache path if needed
        induction_cache_file = None
        if args.fewshot_method == INDUCTION:
            induction_model = args.induction_model_name or args.model_name
            induction_cache_file = get_induction_cache_path(output_dir=args.output_dir, model_name=induction_model, dataset=args.dataset, num_auxiliary_contexts=num_auxiliary_contexts, num_demos=args.induction_num_demos)
            assert os.path.exists(induction_cache_file), f"Induction cache file not found: {induction_cache_file}"

        # ToolUse expects boolean structured_output
        tooluse_structured = args.structured_output != "none"
        train_dataset, eval_dataset = load_tooluse_dataset(
            seed=args.seed, num_auxiliary_contexts=num_auxiliary_contexts,
            structured_output=tooluse_structured,
            fewshot_method=args.fewshot_method,
            retrieval_model=args.retrieval_model,
            output_dir=args.output_dir,
            demo_k=args.demo_k,
            induction_cache_file=induction_cache_file,
        )
        if args.num_train_examples is not None:
            train_dataset = train_dataset.select(range(min(args.num_train_examples, len(train_dataset))))
        if args.num_eval_examples is not None:
            eval_dataset = eval_dataset.select(range(min(args.num_eval_examples, len(eval_dataset))))


    elif args.dataset == SCIENCEQA:
        # Construct induction cache path if needed
        induction_cache_file = None
        if args.fewshot_method == INDUCTION:
            induction_model = args.induction_model_name or args.model_name
            induction_cache_file = get_induction_cache_path(output_dir=args.output_dir, model_name=induction_model, dataset=args.dataset, num_auxiliary_contexts=num_auxiliary_contexts, num_demos=args.induction_num_demos)
            assert os.path.exists(induction_cache_file), f"Induction cache file not found: {induction_cache_file}"

        scienceqa_structured = args.structured_output if args.structured_output != "none" else None
        train_dataset, eval_dataset = load_scienceqa_training_dataset(
            seed=args.seed,
            max_train_samples=args.num_train_examples,
            max_eval_samples=args.num_eval_examples,
            num_auxiliary_contexts=num_auxiliary_contexts,
            fewshot_method=args.fewshot_method,
            structured_output=scienceqa_structured,
            retrieval_model=args.retrieval_model,
            embedding_cache_dir=args.embedding_cache_dir,
            demo_k=args.demo_k,
            induction_cache_file=induction_cache_file,
            subset=args.scienceqa_subset,
            text_only=True,
            include_lecture=args.scienceqa_include_lecture,
        )


    elif args.dataset == COS_E:
        # Construct induction cache path if needed
        induction_cache_file = None
        if args.fewshot_method == INDUCTION:
            induction_model = args.induction_model_name or args.model_name
            induction_cache_file = get_induction_cache_path(output_dir=args.output_dir, model_name=induction_model, dataset=args.dataset, num_auxiliary_contexts=num_auxiliary_contexts, num_demos=args.induction_num_demos)
            assert os.path.exists(induction_cache_file), f"Induction cache file not found: {induction_cache_file}"

        cos_e_structured = args.structured_output if args.structured_output != "none" else None
        train_dataset, eval_dataset = load_cos_e_training_dataset(
            seed=args.seed,
            max_train_samples=args.num_train_examples,
            max_eval_samples=args.num_eval_examples,
            num_auxiliary_contexts=num_auxiliary_contexts,
            fewshot_method=args.fewshot_method,
            structured_output=cos_e_structured,
            retrieval_model=args.retrieval_model,
            embedding_cache_dir=args.embedding_cache_dir,
            demo_k=args.demo_k,
            induction_cache_file=induction_cache_file,
        )


    elif args.dataset == MEDMCQA:
        induction_cache_file = None
        if args.fewshot_method == INDUCTION:
            induction_model = args.induction_model_name or args.model_name
            induction_cache_file = get_induction_cache_path(output_dir=args.output_dir, model_name=induction_model, dataset=args.dataset, num_auxiliary_contexts=num_auxiliary_contexts, num_demos=args.induction_num_demos)
            assert os.path.exists(induction_cache_file), f"Induction cache file not found: {induction_cache_file}"

        medmcqa_structured = args.structured_output if args.structured_output != "none" else None
        train_dataset, eval_dataset = load_medmcqa_training_dataset(
            seed=args.seed,
            max_train_samples=args.num_train_examples,
            max_eval_samples=args.num_eval_examples,
            num_auxiliary_contexts=num_auxiliary_contexts,
            fewshot_method=args.fewshot_method,
            structured_output=medmcqa_structured,
            retrieval_model=args.retrieval_model,
            embedding_cache_dir=args.embedding_cache_dir,
            demo_k=args.demo_k,
            induction_cache_file=induction_cache_file,
        )

    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # Attach bad_teacher_prompt for contrastive loss
    if args.contrastive_weight > 0 or args.mode in {CONTRASTIVE, UNISD_STAR}:
        from src.utils.path_utils import get_neg_demo_cache_path
        cache_dir = args.negative_demo_cache or "outputs/cache/demonstration"
        neg_demo_path = get_neg_demo_cache_path(cache_dir, args.dataset, args.model_name, max_samples=None)
        assert os.path.exists(neg_demo_path), \
            f"Negative demo cache not found: {neg_demo_path}. " \
            f"Generate with: python -m src.teacher.negative_demonstrations --model_name {args.model_name} --dataset {args.dataset}"
        train_dataset = attach_bad_teacher_prompts(
            train_dataset, neg_demo_path, args.dataset, seed=args.seed,
        )

    print(f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")

    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, trust_remote_code=True)
    teacher_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    disable_thinking_mode(tokenizer)
    tokenizer.padding_side = "left"

    # ── Prompt length sanity check — drop overflow examples ─────
    overflow_indices = set()
    max_seen = 0

    for i in range(len(train_dataset)):
        tp = train_dataset[i]["teacher_prompt"]
        n_tok = len(tokenizer.apply_chat_template(tp, add_generation_prompt=True))
        max_seen = max(max_seen, n_tok)
        if n_tok > args.max_prompt_length:
            overflow_indices.add(i)
        # Also check fewshot teacher prompts
        for k in range(num_auxiliary_contexts):
            key = f"fewshot_teacher_prompt_{k}"
            if key in train_dataset.column_names:
                fp = train_dataset[i][key]
                n_tok_fs = len(tokenizer.apply_chat_template(fp, add_generation_prompt=True))
                max_seen = max(max_seen, n_tok_fs)
                if n_tok_fs > args.max_prompt_length:
                    overflow_indices.add(i)
        # Also check bad_teacher_prompt (contrastive negative demo)
        if "bad_teacher_prompt" in train_dataset.column_names:
            btp = train_dataset[i]["bad_teacher_prompt"]
            if btp:
                n_tok_bad = len(tokenizer.apply_chat_template(btp, add_generation_prompt=True))
                max_seen = max(max_seen, n_tok_bad)
                if n_tok_bad > args.max_prompt_length:
                    overflow_indices.add(i)

    if overflow_indices:
        pct = len(overflow_indices) / len(train_dataset) * 100
        keep_indices = [i for i in range(len(train_dataset)) if i not in overflow_indices]
        train_dataset = train_dataset.select(keep_indices)
        print(f"[Prompt check] Dropped {len(overflow_indices)} examples ({pct:.1f}%) "
              f"exceeding max_prompt_length={args.max_prompt_length} (max seen: {max_seen} tokens). "
              f"Remaining: {len(train_dataset)}")
    else:
        print(f"[Prompt check] All {len(train_dataset)} teacher_prompts fit within "
              f"max_prompt_length={args.max_prompt_length} (max seen: {max_seen} tokens)")


    # Determine subset for output directory (matches eval path convention)
    if args.dataset == SCIENCEQA:
        dataset_subset = args.scienceqa_subset  # e.g. "all" → dir "scienceqa-all"
    else:
        dataset_subset = None

    hparam_tag = build_hparam_tag(
        args.mode,
        gamma_agreement=args.gamma_agreement,
        num_auxiliary_contexts=args.num_auxiliary_contexts,
        contrastive_weight=args.contrastive_weight,
        contrastive_margin=args.contrastive_margin,
        ref_model_sync_steps=args.ref_model_sync_steps,
        ref_model_mixup_beta=args.ref_model_mixup_beta,
        final_layer_distill_weight=args.final_layer_distill_weight,
        fewshot_method=args.fewshot_method,
        token_clip=args.token_clip,
        token_clip_quantile=getattr(args, "token_clip_quantile", None),
        alpha=args.alpha,
    )
    max_samples_tag = args.num_train_examples if args.num_train_examples < 1_000_000 else None
    args.model_dir = get_model_dir(output_dir=args.output_dir, method=args.mode, dataset=args.dataset, model_name=args.model_name, subset=dataset_subset, hparam_tag=hparam_tag, max_samples=max_samples_tag)

    # Compute max_steps from additional_epochs when resuming from checkpoint
    if args.resume_from_checkpoint and args.additional_epochs > 0:
        import math
        state_path = os.path.join(args.resume_from_checkpoint, "trainer_state.json")
        with open(state_path) as f:
            saved_state = json.load(f)
        saved_step = saved_state["global_step"]
        steps_per_epoch = math.ceil(
            len(train_dataset) / (args.per_device_train_batch_size * args.gradient_accumulation_steps)
        )
        additional_steps = round(args.additional_epochs * steps_per_epoch)
        args.max_steps = saved_step + additional_steps
        print(f"[Resume] checkpoint at step {saved_step}, "
              f"{args.additional_epochs} additional epochs = {additional_steps} steps, "
              f"total max_steps = {args.max_steps}")

    # Teacher EMA mode reuses the existing ref_model sync callback
    if args.mode in {EMA, UNISD_STAR}:
        sync_ref = True
        ref_model_sync_steps = args.ref_model_sync_steps
        ref_model_mixup_beta = args.ref_model_mixup_beta
    else:
        sync_ref = False
        ref_model_sync_steps = 512   # default, unused
        ref_model_mixup_beta = 0.4   # default, unused

    config = UniSDConfig(
        seed=args.seed,
        # Generation backend
        use_vllm=args.use_vllm,
        vllm_mode="colocate",
        vllm_tensor_parallel_size=1,
        vllm_gpu_memory_utilization=getattr(args, "vllm_gpu_memory_utilization", 0.3),
        vllm_enable_sleep_mode=getattr(args, "vllm_enable_sleep_mode", True),
        # Optimizer / scheduleƒƒ
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=1,
        bf16=True,
        fp16=False,
        # Batching
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_generations=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        # Training length
        num_train_epochs=1,
        max_steps=args.max_steps,
        # Checkpointingƒ
        save_total_limit=args.save_total_limit,
        max_grad_norm=1,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Logging
        report_to="none",
        output_dir=args.model_dir,
        log_completions=False,
        eval_strategy="no",
        save_strategy="epoch",
        # Ref model sync (used by teacher_ema mode)
        sync_ref_model=sync_ref,
        ref_model_sync_steps=ref_model_sync_steps,
        ref_model_mixup_beta=ref_model_mixup_beta,
        vllm_importance_sampling_correction=False,
        num_loss_tokens_to_skip=0,
        # Over-generation control
        mask_truncated_completions=args.mask_truncated_completions,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        # DeepSpeed
        deepspeed=args.deepspeed,
        # Mode and agreement args
        mode=args.mode,
        num_auxiliary_contexts=args.num_auxiliary_contexts,
        gamma_agreement=args.gamma_agreement,
        agreement_stat=args.agreement_stat,
        # Context augmentation
        context_augment=args.context_augment,
        context_drop_prob=args.context_drop_prob,
        # Contrastive loss
        contrastive_weight=args.contrastive_weight,
        contrastive_margin=args.contrastive_margin,
        # Final-layer hidden-state distillation
        final_layer_distill_weight=args.final_layer_distill_weight,
        # JSD interpolation (0=forward KL, 1=reverse KL, in between=JSD)
        alpha=args.alpha,
        # Per-token JSD divergence clipping (static or adaptive)
        token_clip=args.token_clip,
        token_clip_quantile=args.token_clip_quantile,
    )

    peft_config = get_default_lora_config(args.model_name)
    trainer = UniSDTrainer(
        model=model,
        ref_model=teacher_model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    os.makedirs(args.model_dir, exist_ok=True)

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    peak_gb = None
    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"Peak GPU memory allocated: {peak_gb:.2f} GB")

    print(f"\nMulti-teacher training completed successfully! Output dir: {args.model_dir}")

    try:
        log_history = getattr(getattr(trainer, "state", None), "log_history", None)
        train_metrics = extract_final_train_metrics(log_history)
        train_loss = train_metrics.get("train_loss")
        if isinstance(train_loss, (int, float)) and math.isfinite(train_loss):
            train_metrics["train_ppl"] = math.exp(train_loss)
        total_tokens = train_metrics.pop("num_tokens", None)
        if isinstance(total_tokens, float):
            total_tokens = int(total_tokens)

        record = compute_resource_consumption_record(
            method=getattr(args, "mode", "multi_teacher"),
            model=getattr(args, "model_name", "unknown"),
            dataset=getattr(args, "dataset", "unknown"),
            phase="train",
            wall_time_sec=time.time() - t0,
            num_gpus=detect_num_gpus(),
            peak_gpu_mem_gb=peak_gb,
            total_tokens=total_tokens,
            subset=getattr(args, "scienceqa_subset", None),
            num_train_examples=getattr(args, "num_train_examples", None),
            num_auxiliary_contexts=getattr(args, "num_auxiliary_contexts", None),
            **train_metrics,
        )
        append_resource_record(record)
    except Exception as e:
        print(f"[resource_consumption] WARNING: failed to log record: {e}")


if __name__ == "__main__":
    main()
