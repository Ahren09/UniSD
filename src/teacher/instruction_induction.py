"""Instruction induction for teacher context generation.

Uses vLLM offline LLM class for batched generation of induced instructions
from demo examples. Results are cached to disk for reuse.
"""

import hashlib
import json
import os
from pathlib import Path
from random import Random
import random
import json

from pydantic import BaseModel
from vllm import LLM, SamplingParams

from src.utils.env_utils import get_model_dtype  # noqa: F401 — env_utils import patches transformers for custom-code models

from src.train.train_args import parse_args
from src.data.mbpp import load_mbpp, build_code_generation_prompt, _fix_function_name_in_code
from src.const import *
from src.utils.path_utils import get_induction_cache_path
from src.data.scienceqa import load_scienceqa
from src.prompts.mcqa import build_mcqa_prompt
from src.data.gpqa import load_gpqa



class InductionResponse(BaseModel):
    instruction: str
    rationale: str

PLAIN_RESPONSE = "Output ONLY the instruction text, nothing else:"
STRUCTURED_RESPONSE = (
    'Respond with ONLY a JSON object (no markdown fences, no extra text):\n'
    '{"instruction": "<3-4 sentence instruction>", "rationale": "<brief reason why this helps>"}'
)


def build_induction_meta_prompt(demos: list[dict], dataset_name: str, structured_output: bool = False) -> str:
    """Build a meta-prompt that asks an LLM to induce a coding instruction from demos.
    
    Sample: 
        Solve the given programming task by producing exactly one Python function in a single python block. Ensure the function name and signature exactly match the specification and assertions, include any required imports, and return only the requested implementation without extra text or test code.

    Args:
        demos: List of dicts with keys "prompt", "code", and optionally "function_name".
        dataset_name: Name of the dataset being used.
    """
    
    if dataset_name == MBPP:
        examples = []
        for idx, demo in enumerate(demos, 1):
            
            test_cases = '\n'.join(demo.get("test_list", []))
            examples.append(f"""### Example {idx}
{demo["prompt"]}

Test Cases:
{test_cases}

Solution:
```python
{demo["code"]}
```
-----
""")

        examples_text = "\n\n".join(examples)

        prompt = f"""You are given several programming tasks and their correct Python solutions. 
        
## Examples
The following are reference examples. Do NOT solve them.
-----
{examples_text}

## Your Task now

Your goal is to give me ONE concise, high-level instruction that would help a programmer solve similar coding tasks correctly.

## Rules
- Focus only on general coding/output rules.
- Do NOT mention any example-specific names, values, or task details.
- Do NOT output code.
- Do NOT output a ```python ... ``` block.
- Write only 3-4 sentence.

{STRUCTURED_RESPONSE if structured_output else PLAIN_RESPONSE}
"""

    elif dataset_name in {GPQA}:
        examples = []
        for idx, demo in enumerate(demos, 1):
            options_text = "\n".join(
                f"{k}) {v}" for k, v in sorted(demo["options"].items())
            )
            examples.append(f"""### Example {idx}
Question: {demo["question"]}

{options_text}

Correct Answer: {demo["correct_letter"]}
Explanation: {demo.get("explanation", "N/A")}
-----
""")

        examples_text = "\n\n".join(examples)

        prompt = f"""You are given several questions and their correct answers with explanations.

## Examples
The following are reference examples. Do NOT answer them.
-----
{examples_text}

## Your Task now

Your goal is to give me ONE concise, high-level instruction that would help someone answer similar questions correctly.

## Rules
- Focus on reasoning/answering rules.
- Do NOT mention any example-specific questions, answers, or domain details.
- Do NOT output an answer choice.
- Write only 3-4 sentences.

{STRUCTURED_RESPONSE if structured_output else PLAIN_RESPONSE}
"""

    elif dataset_name in {SCIENCEQA, COS_E, MEDMCQA}:
        examples = []
        for idx, demo in enumerate(demos, 1):
            options_text = "\n".join(
                f"{k}) {v}" for k, v in sorted(demo["options"].items())
            )
            examples.append(f"""### Example {idx}
Question: {demo["question"]}

{options_text}

Correct Answer: {demo["correct_letter"]}
Explanation: {demo.get("explanation", "N/A")}
-----
""")

        examples_text = "\n\n".join(examples)

        prompt = f"""You are given several multiple-choice questions and their correct answers with explanations.

## Examples
The following are reference examples. Do NOT answer them.
-----
{examples_text}

## Your Task now

Your goal is to give me ONE concise, high-level instruction that would help someone answer similar multiple-choice questions correctly.

## Rules
- Focus only on general reasoning/answering rules.
- Do NOT mention any example-specific questions, answers, or domain details.
- Do NOT output an answer choice.
- Write only 3-4 sentences.

{STRUCTURED_RESPONSE if structured_output else PLAIN_RESPONSE}
"""

    elif dataset_name == TOOLUSE:
        examples = []
        for idx, demo in enumerate(demos, 1):
            golden_text = '\n'.join(demo['golden_response']) if isinstance(demo['golden_response'], list) else demo['golden_response']
            examples.append(f"""### Example {idx}
{demo["instruction"]}

Response:
{golden_text}
-----
""")

        examples_text = "\n\n".join(examples)

        prompt = f"""You are given several tool-use tasks and their correct responses (in ReAct format with Thought/Action/Action_Input).

## Examples
The following are reference examples. Do NOT solve them.
-----
{examples_text}

## Your Task now

Your goal is to give me ONE concise, high-level instruction that would help an assistant solve similar tool-use tasks correctly.

## Rules
- Focus only on general reasoning/tool-use rules.
- Do NOT mention any example-specific tool names, URLs, or parameter values.
- Do NOT output a tool call or Action/Action_Input.
- Write only 3-4 sentences.

{STRUCTURED_RESPONSE if structured_output else PLAIN_RESPONSE}
"""

    else:
        raise ValueError(f"Unsupported dataset for induction: {dataset_name}")

    return prompt



def induce_instructions_batch(
    demos_batch: list[list[dict]],
    llm: LLM,
    dataset_name: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    max_retries: int = 2,
    structured_output: bool = False,
) -> list[dict] | list[str]:
    """Generate induced instructions for a batch of demo sets using vLLM.

    Args:
        structured_output: If True, request JSON output with instruction/rationale
            fields, validate with Pydantic, and retry failures. Returns list[dict].
            If False (default), return plain instruction strings. Returns list[str].
    """
    prompts = [build_induction_meta_prompt(demos, dataset_name, structured_output=structured_output)
               for demos in demos_batch]
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_new_tokens)

    if not structured_output:
        # Plain text mode: single pass, return strings
        print(f"[Induction] Generating {len(prompts)} instructions (plain text)...")
        outputs = llm.generate(prompts, sampling_params)
        results = [o.outputs[0].text.strip() for o in outputs]
        print(f"[Induction] Generated {len(results)} instructions "
              f"(avg len: {sum(len(r) for r in results) / len(results):.0f} chars)")
        return results

    # Structured JSON mode: generate + parse + retry loop
    results: list[dict | None] = [None] * len(prompts)
    pending = list(range(len(prompts)))

    for attempt in range(max_retries + 1):
        outputs = llm.generate([prompts[i] for i in pending], sampling_params)
        still_pending = []
        for j, orig_idx in enumerate(pending):
            text = outputs[j].outputs[0].text.strip()
            try:
                parsed = InductionResponse.model_validate_json(text)
                results[orig_idx] = parsed.model_dump()
            except Exception:
                still_pending.append(orig_idx)
        print(f"[Induction] Attempt {attempt+1}: {len(pending)-len(still_pending)}/{len(pending)} valid JSON")
        pending = still_pending
        if not pending:
            break

    # Fallback: empty dict for anything still failing
    for orig_idx in pending:
        results[orig_idx] = {"instruction": "", "rationale": ""}
    if pending:
        print(f"[Induction] WARNING: {len(pending)} instructions failed all retries, using empty fallback")

    return results



def load_or_induce_instructions(
    train_problems: list[dict],
    train_data: list[dict],
    num_contexts: int,
    num_demos: int,
    demo_select_strategy: str,
    model_name: str,
    temperature: float,
    max_new_tokens: int,
    cache_dir: str,
    dataset_name: str,
    seed: int = 42,
    retrieval_model: str | None = None,
    embedding_cache_dir: str | None = None,
    structured_output: bool = False,
) -> list:
    """Load cached or generate induced instructions for all training samples.

    Args:
        train_problems: Raw MBPP problems (with test_list, code, etc).
        train_data: Formatted training data (with prompt, teacher_prompt, _fixed_code).
        num_contexts: Number of teacher contexts (K).
        num_demos: Number of demo pairs per induction meta-prompt.
        demo_select_strategy: "random" or "retrieval" for demo selection.
        model_name: Model for instruction induction.
        temperature: Sampling temperature for induction.
        max_new_tokens: Max tokens per induced instruction.
        cache_dir: Directory for caching results.
        seed: Random seed.
        retrieval_model: Sentence-transformers model (for retrieval strategy).
        embedding_cache_dir: Cache dir for embeddings (for retrieval strategy).
        structured_output: If True, return dicts with instruction/rationale.
            If False, return plain instruction strings.

    Returns:
        Flat list of instructions (dicts if structured_output=True, strings if False).
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    cache_file = get_induction_cache_path(output_dir=cache_dir, model_name=model_name, dataset=dataset_name, num_auxiliary_contexts=num_contexts, num_demos=num_demos)

    # Check cache
    if cache_file.exists():
        print(f"[Induction] Loading cached instructions from {cache_file}")
        with open(cache_file) as f:
            instructions = json.load(f)
        print(f"[Induction] Loaded {len(instructions)} cached instructions")
        return instructions

    N = len(train_data)
    
    # Max number of unique demo sets, each creating a separate induction instruction. 
    max_num_meta_instructions = len(train_data) // num_demos
    num_meta_instructions = min(max_num_meta_instructions, 100)
    
    random_indices = random.sample(range(N), N)[:num_demos * num_meta_instructions]
    
    demo_indices = [random_indices[i * num_demos: (i + 1) * num_demos] for i in range(num_meta_instructions)]
    assert len(demo_indices) == num_meta_instructions and all(len(d) == num_demos for d in demo_indices)
    
    demos = [[train_problems[i] for i in indices] for indices in demo_indices]
    
    
    # Create vLLM instance once for all induction
    llm = LLM(model=model_name, gpu_memory_utilization=0.85, dtype=get_model_dtype(), trust_remote_code=True)

    # Generate instructions in one vLLM batch
    instructions = induce_instructions_batch(
        demos, llm, dataset_name=dataset_name, max_new_tokens=max_new_tokens,
        temperature=temperature, structured_output=structured_output,
    )
    
    os.makedirs(cache_file.parent, exist_ok=True)
    if not os.path.exists(cache_file):
        with open(cache_file, "w") as f:
            json.dump(instructions, f, indent=2)
        
        print(f"[Induction] Saved {len(instructions)} instruction sets to {cache_file}")

    return instructions


if __name__ == "__main__":
    args = parse_args()

    num_contexts = args.num_auxiliary_contexts or 3
    num_demos = args.induction_num_demos
    cache_dir = args.output_dir

    dataset_name = getattr(args, "dataset", MBPP)

    if dataset_name == TOOLUSE:
        from datasets import load_dataset
        # Load raw ToolUse data from HuggingFace (Ahren09/ToolAlpaca)
        raw_data = load_dataset("Ahren09/ToolAlpaca", split="train").to_list()
        # train_problems = raw data dicts (with instruction, golden_response, prompt)
        train_problems = raw_data
        # train_data: minimal formatted list (just needs prompt for pipeline compat)
        train_data = [
            {"prompt": [{"role": "user", "content": ex["prompt"]}]}
            for ex in raw_data
        ]
    elif dataset_name == GPQA:
    
        gpqa_subset = getattr(args, "gpqa_subset", "gpqa_extended")
        train_problems = load_gpqa(gpqa_subset, seed=args.seed)
        train_data = []
        for p in train_problems:
            student_text = build_mcqa_prompt(p)
            train_data.append({
                "prompt": [{"role": "user", "content": student_text}],
            })
    elif dataset_name == SCIENCEQA:


        scienceqa_subset = getattr(args, "scienceqa_subset", "all")
        include_lecture = getattr(args, "scienceqa_include_lecture", True)
        train_problems = load_scienceqa("train", seed=args.seed, subset=scienceqa_subset)
        train_data = []
        for p in train_problems:
            context = p.get("lecture") if include_lecture and p.get("lecture") else None
            student_text = build_mcqa_prompt(p, context=context)
            train_data.append({
                "prompt": [{"role": "user", "content": student_text}],
            })
    elif dataset_name == COS_E:
        from src.data.cos_e import load_cos_e
        train_problems = load_cos_e("train", seed=args.seed)
        train_data = []
        for p in train_problems:
            student_text = build_mcqa_prompt(p)
            train_data.append({
                "prompt": [{"role": "user", "content": student_text}],
            })
    elif dataset_name == MEDMCQA:
        from src.data.medmcqa import load_medmcqa
        train_problems = load_medmcqa("train", seed=args.seed)
        train_data = []
        for p in train_problems:
            student_text = build_mcqa_prompt(p)
            train_data.append({
                "prompt": [{"role": "user", "content": student_text}],
            })
    else:
        # MBPP (default)
        train_problems = load_mbpp("train", config=args.mbpp_config)
        train_data = []
        for p in train_problems:
            student_text = build_code_generation_prompt(p, "mbpp")
            fixed_code = _fix_function_name_in_code(
                p["code"], p.get("code_function_name"), p.get("test_function_name")
            )
            train_data.append({
                "prompt": [{"role": "user", "content": student_text}],
                "_fixed_code": fixed_code,
            })

    # Generate and cache instructions
    instructions = load_or_induce_instructions(
        train_problems, train_data, num_contexts, num_demos,
        args.demo_select_strategy, args.model_name, args.induction_temperature,
        args.induction_max_new_tokens, cache_dir, dataset_name=dataset_name, seed=args.seed,
        retrieval_model=args.retrieval_model, embedding_cache_dir=args.embedding_cache_dir,
        structured_output=(args.structured_output != "none"),
    )

    
   