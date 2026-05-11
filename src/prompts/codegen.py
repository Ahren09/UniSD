"""Prompt builder for code generation tasks."""

import ast


def build_code_generation_prompt(problem: dict, dataset_name: str) -> str:
    """Build prompt asking for Python code only.

    Args:
        problem: dict with at least 'prompt' key
        dataset_name: 'mbpp' or 'humaneval'
    """
    
    if dataset_name == "mbpp":
        
        tree = ast.parse(problem["code"])
        func_def = next((node for node in tree.body if isinstance(node, ast.FunctionDef)), None)
        assert func_def is not None, "No function definition found."

        arg_names = [arg.arg for arg in func_def.args.args]
        function_signature = f"{func_def.name}({', '.join(arg_names)})"
        
        test_cases = '\n'.join(problem['test_list'])
        
        prompt = f"""{problem['prompt']}
The required function signature is: {function_signature}

## Note
- Write only the function implementation. Do not include any extra texts, explanations, test code, or examples.
- Only return the code within a ```python ... ``` block.
- Make sure the function name is correct as per the assertions.
- Include necessary imports.

## Answer Format
```python
def {function_signature}:
    ...
```

## Test Cases
{test_cases}
"""

    elif dataset_name == "humaneval":
        prompt = f"""{problem['prompt']}

## Note
1. Return the COMPLETE function, including the original function signature and docstring.
2. Include all necessary imports (e.g., math, collections, typing) inside the code block.
3. Only return the Python code within a ```python ... ``` markdown block.
4. Do not provide any explanations, comments (unless inside the code), or usage examples.
5. Ensure the function name and parameters match the provided signature exactly.
"""

    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")
    
    
    return prompt
