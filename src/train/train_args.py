import argparse
from src.const import *


def parse_args(argv=None):
    """Shared argument parser for training and induction scripts."""
    parser = argparse.ArgumentParser(description="UniSD Entry")
    # Mode
    parser.add_argument("--mode", type=str, default=UNISD_STAR,
                        choices=sorted(ALL_MODES),
                        help="UniSD training mode: agreement_seq_random, agreement_seq_retrieval, "
                             "agreement_seq_induction, contrastive, ema, match_joint, match_repr, "
                             "unisd_star (default), clip, agreement_tok_random, "
                             "agreement_tok_retrieval, agreement_tok_induction")
    parser.add_argument("--fewshot_method", type=str, default=None,
                        choices=sorted(FEWSHOT_METHODS),
                        help="Fewshot context-selection strategy (random/retrieval/induction). "
                             "Required for --mode=unisd_star; auto-derived for agreement modes.")
    parser.add_argument("--dataset", type=str, default="tooluse",
                        choices=["tooluse", "mbpp", "humaneval", "gpqa", "scienceqa", "cos_e", "medmcqa", "gsm8k"],
                        help="Training dataset: tooluse (default), mbpp, humaneval, gpqa, scienceqa, cos_e, medmcqa, or gsm8k")
    parser.add_argument("--mbpp_config", type=str, default="sanitized",
                        choices=["sanitized", "full"],
                        help="MBPP dataset config: sanitized (120 train) or full (374 train)")
    parser.add_argument("--gpqa_subset", type=str, default="gpqa_extended",
                        choices=["gpqa_main", "gpqa_diamond", "gpqa_extended"],
                        help="GPQA dataset subset")
    parser.add_argument("--scienceqa_subset", type=str, default="all",
                        choices=["all", "natural_science", "language_science", "social_science"],
                        help="ScienceQA subject filter")
    parser.add_argument("--scienceqa_include_lecture", action=argparse.BooleanOptionalAction, default=True,
                        help="Include lecture text as context in ScienceQA prompts")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output_dir", type=str, default="outputs/unisd/")
    parser.add_argument("--max_steps", type=int, default=-1, help="Max training steps (-1 = train full epochs)")
    parser.add_argument("--num_train_examples", type=int, default=1_000_000)
    parser.add_argument("--num_eval_examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_completion_length", type=int, default=2048)
    parser.add_argument("--use_vllm", action="store_true", default=False)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.3,
                        help="Fraction of GPU memory reserved for vLLM (default 0.3)")
    parser.add_argument("--vllm_enable_sleep_mode", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable vLLM sleep-mode between train/generate phases. "
                             "Disable (--no-vllm_enable_sleep_mode) on drivers without cuMem sleep support.")
    parser.add_argument("--deepspeed", type=str, default=None, help="Path to DeepSpeed config JSON")
    # Checkpoint resume
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint dir to resume training from")
    parser.add_argument("--additional_epochs", type=float, default=0,
                        help="Additional epochs to train beyond checkpoint (e.g. 0.2). "
                             "Only used with --resume_from_checkpoint.")
    # Training hyperparameters
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_generations", type=int, default=1)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--save_total_limit", type=int, default=2)
    # Generation / over-generation control
    parser.add_argument("--mask_truncated_completions", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    # Teacher count (fewshot count for fewshot modes)
    parser.add_argument("--num-auxiliary-contexts", type=int, default=0,
                        help="Number of auxiliary contexts (fewshot prompts or augmented contexts)")
    # Agreement weighting args
    parser.add_argument("--gamma_agreement", type=float, default=None,
                        help="Agreement weighting strength (None = mode default)")
    parser.add_argument("--agreement_stat", type=str, default="var", choices=["var", "range"])
    parser.add_argument("--log_agreement_stats", action=argparse.BooleanOptionalAction, default=True)
    # Context augmentation (hidden override)
    parser.add_argument("--context_augment", type=str, default=None,
                        choices=["none", "drop_demo", "token_drop"],
                        help="Context augmentation strategy (default: derived from mode)")
    parser.add_argument("--context_drop_prob", type=float, default=0.5,
                        help="Drop probability for token_drop augmentation")
    # Structured output
    parser.add_argument("--structured_output", type=str, default="none",
                        choices=["none", "letter", "json"],
                        help="Structured output mode: none (default), letter, json")
    # Retrieval args
    parser.add_argument("--retrieval-model", type=str, default="all-MiniLM-L6-v2",
                        help="Sentence-transformers model for embedding-based retrieval")
    parser.add_argument("--embedding_cache_dir", type=str, default="data/.cache",
                        help="Cache directory for sentence embeddings")
    parser.add_argument("--demo_k", type=int, default=1,
                        help="Number of retrieved demos per fewshot teacher context")
    # Induction args
    parser.add_argument("--demo_select_strategy", type=str, default="random",
                        choices=["random", "retrieval"],
                        help="How to select demos for induction meta-prompt")
    parser.add_argument("--induction_num_demos", type=int, default=5,
                        help="Number of demo pairs per induction meta-prompt")
    parser.add_argument("--induction_temperature", type=float, default=0.7,
                        help="Temperature for instruction induction generation")
    parser.add_argument("--induction_max_new_tokens", type=int, default=256,
                        help="Max tokens per induced instruction")
    parser.add_argument("--induction_model_name", type=str, default=None,
                        help="Model for instruction induction (defaults to --model_name)")
    # Contrastive loss args
    parser.add_argument("--contrastive_weight", type=float, default=0.0,
                        help="Weight for bad-teacher contrastive loss (0 = disabled)")
    parser.add_argument("--contrastive_margin", type=float, default=0.5,
                        help="Margin for contrastive hinge loss")
    parser.add_argument("--negative_demo_cache", type=str, default=None,
                        help="Cache directory for negative demos (default: outputs/cache/demonstration)")
    # Teacher EMA args
    parser.add_argument("--ref_model_sync_steps", type=int, default=10,
                        help="Steps between teacher EMA syncs (only for --mode ema)")
    parser.add_argument("--ref_model_mixup_beta", type=float, default=0.9,
                        help="EMA interpolation weight: teacher = beta*snapshot + (1-beta)*student. Larger beta means more emphasis on the previous snapshot (slower EMA), smaller beta follows the student more closely.")
    # Final-layer hidden-state distillation
    parser.add_argument("--final_layer_distill_weight", type=float, default=0.1,
                        help="Weight for auxiliary final-layer hidden-state MSE loss (0 = disabled)")
    # JSD interpolation between forward KL (alpha=0) and reverse KL (alpha=1)
    parser.add_argument("--alpha", type=float, default=0.0,
                        help="JSD interpolation: 0.0=forward KL (default), 1.0=reverse KL, "
                             "in between=Jensen-Shannon Divergence. Used by clip mode to choose "
                             "the divergence between student and teacher.")
    # Per-token KL/JSD clipping (used by --mode=clip)
    parser.add_argument("--token_clip", type=float, default=None,
                        help="Per-token KL/JSD clip ceiling, static (e.g. 10.0). "
                             "None (default) = no clipping. Mutually exclusive with "
                             "--token_clip_quantile.")
    parser.add_argument("--token_clip_quantile", type=float, default=None,
                        help="Per-token KL/JSD clip ceiling, adaptive: empirical quantile "
                             "(e.g. 0.99) of masked completion-token losses per batch. "
                             "Mutually exclusive with --token_clip.")
    
    parser.add_argument("--max_seq_length", type=int, default=1024,
                        help="Max sequence length for training")
    parser.add_argument("--num_train_epochs", type=int, default=1,
                        help="Number of training epochs.")
    parser.add_argument("--logging_steps", type=int, default=5,
                        help="Logging interval for training")
    parser.add_argument("--save_steps", type=int, default=50,
                        help="Checkpoint save interval for training")
    parser.add_argument("--reasoning", action="store_true", default=False,
                        help="Enable reasoning for training")
    return parser.parse_args(argv)


def apply_mode_defaults(args):
    """Fill None fields based on --mode. Call right after parse_args().

    Sets gamma_agreement and context_augment defaults per mode.
    For fewshot modes, num_auxiliary_contexts must be >0.
    """
    mode = args.mode

    if mode in (CONTRASTIVE, EMA, MATCH_JOINT, MATCH_REPR):
        if args.gamma_agreement is None:
            args.gamma_agreement = 0.0
        if args.num_auxiliary_contexts == 0:
            pass  # token_cl use no extra contexts
        if mode == CONTRASTIVE and args.contrastive_weight <= 0:
            args.contrastive_weight = 0.1  # sensible default for token_cl

    elif mode == CLIP:
        # per-token KL clipping in the loss.
        if args.gamma_agreement is None:
            args.gamma_agreement = 0.0
        if args.token_clip is not None and args.token_clip_quantile is not None:
            raise ValueError(
                "--token_clip and --token_clip_quantile are mutually exclusive; "
                "set exactly one."
            )
        if args.token_clip is None and args.token_clip_quantile is None:
            args.token_clip = 10.0  # default: caps real outliers only

    elif mode == UNISD_STAR:
        # UNISD_STAR combines fewshot agreement + contrastive + EMA + final-layer distillation
        if args.gamma_agreement is None:
            args.gamma_agreement = 1.0
        if args.contrastive_weight <= 0:
            args.contrastive_weight = 0.1
        if args.num_auxiliary_contexts == 0:
            raise ValueError(f"--mode={mode} requires --num-auxiliary-contexts > 0")
        if args.fewshot_method is None:
            raise ValueError(f"--mode={mode} requires --fewshot_method (random/retrieval/induction)")

    elif mode in AGREEMENT_MODES:
        if args.gamma_agreement is None:
            args.gamma_agreement = 1.0
        if args.num_auxiliary_contexts == 0:
            raise ValueError(f"--mode={mode} requires --num-auxiliary-contexts > 0")
        args.fewshot_method = MODE_TO_FEWSHOT[mode]

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Default context_augment to "none" if still unset
    if args.context_augment is None:
        args.context_augment = "none"

    return args
