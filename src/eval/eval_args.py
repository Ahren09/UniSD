import argparse
import os


def add_common_eval_args(parser, max_new_tokens_default=1024):
    """Add evaluation arguments shared across all eval scripts."""
    parser.add_argument("--mode", type=str, required=True,
                        help="Experiment mode (e.g., unisd_star, token_cl, agreement_tok_retrieval)")
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Path to model checkpoint or HuggingFace model ID")
    parser.add_argument("--max_new_tokens", type=int, default=max_new_tokens_default,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum number of samples to evaluate")
    parser.add_argument("--max_prompt_length", type=int, default=1024,
                        help="Maximum prompt length in tokens")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Directory to write evaluation results")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature for generation")
    parser.add_argument("--vllm_mode", type=str, choices=["colocate", "server"], default="colocate",
                        help="vLLM backend: 'colocate' for local LLM, 'server' for external OpenAI-compatible API")
    parser.add_argument("--vllm_server_url", type=str, default="http://localhost:8000",
                        help="vLLM server URL (only used in server mode)")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Tensor parallel size (only used in colocate mode)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                        help="GPU memory utilization (only used in colocate mode)")
    parser.add_argument("--force_remerge", action="store_true", default=False,
                        help="Force re-merge of LoRA adapter even if cached merge exists")
    parser.add_argument("--hparam-tag", type=str, default="",
                        help="Hyperparameter subdirectory tag (e.g. gamma1.0_nctx3)")
    parser.add_argument("--suffix", type=str, default="",
                        help="Suffix to append to dataset tag in output path (e.g. max1000)")
    parser.add_argument("--enforce_eager", action="store_true", default=False,
                        help="Disable CUDA graph capture in vLLM (needed for V100 / OOM during graph capture)")
    # Retention-to-base scoring (avg log p_base(y_ft | x))
    parser.add_argument("--compute_retention", action=argparse.BooleanOptionalAction, default=True,
                        help="After generation, score completions under the BASE model with HF teacher forcing "
                             "and report avg log p_base(y_ft | x). Loads the base separately; the FT model is "
                             "not used for this pass. Pass --no-compute_retention to disable.")
    parser.add_argument("--base_model_name_or_path", type=str, default=None,
                        help="Path or HF id of the base model used for retention scoring. Required when "
                             "--compute_retention is set and --model_name_or_path is not a LoRA adapter "
                             "(LoRA checkpoints auto-resolve their base via adapter_config.json).")
    parser.add_argument("--retention_batch_size", type=int, default=4,
                        help="Mini-batch size for the HF forward pass during retention scoring. "
                             "Both models (base + FT) are resident on the same GPU; a smaller batch "
                             "(e.g. 1 or 2) is recommended for large models.")
    parser.add_argument("--gpu_memory_fraction", type=float, default=None,
                        help="Fraction of GPU memory to allow PyTorch to use (0.0-1.0). "
                             "Applied via torch.cuda.set_per_process_memory_fraction before model loading. "
                             "Useful for sharing a GPU with other processes.")
    parser.add_argument("--retention_max_length", type=int, default=None,
                        help="Truncation budget for prompt+completion during retention scoring. "
                             "Defaults to max_prompt_length + max_new_tokens. Truncation removes prompt tokens "
                             "from the front; the completion is never truncated.")
    parser.add_argument("--ft_generations_path", type=str, default=None,
                        help="Path to JSONL with cached FT model generations (must have 'task_id' and "
                             "'generated_text' fields). When set, skips vLLM generation and loads from cache.")
    parser.add_argument("--base_generations_path", type=str, default=None,
                        help="Path to JSONL with cached base model generations (must have 'task_id' and "
                             "'generated_text' fields). When set alongside --compute_retention, enables "
                             "reverse-direction scoring: log p_ft(y_base | x) and JSD on base completions.")


def parse_args():
    """Parse args for tool-use evaluation."""
    parser = argparse.ArgumentParser(description="Tool-use evaluation for trained checkpoints")
    add_common_eval_args(parser)
    parser.add_argument("--structured_output", type=bool, default=True,
                        help="Use vLLM guided decoding for JSON output (colocate mode only)")
    parser.add_argument("--max_retries", type=int, default=2,
                        help="Max retries for examples where parsing fails")
    parser.add_argument("--dataset", type=str, required=True, choices=["mbpp", "humaneval", "scienceqa", "gpqa", "cos_e", "tooluse"])
    parser.add_argument("--split", type=str, default="test", choices=["test", "validation", "val", "all"])
    args = parser.parse_args()
    return args
