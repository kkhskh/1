# model_backend.py
from typing import List, Tuple
import requests
import re
from config import MODEL_NAME, MAX_TOKENS, TEMPERATURE
from tools import SolutionCandidate, extract_final_answer, execute_python_blocks

# Use local Ollama API
OLLAMA_BASE_URL = "http://localhost:11434"

PROMPT_TEMPLATES = [
    # Template 0: Concise algebraic approach
    "Solve this math problem step by step. Focus on algebraic manipulation. "
    "If computation is needed, write exactly one fenced python block like:\n\n```python\nRESULT = <expression>\n```\n\n"
    "No imports. No loops. No print. Then continue reasoning using the computed RESULT.\n"
    "End with exactly: FINAL: <integer>\n\nProblem:\n{problem}\n",

    # Template 1: Geometric/structure-first
    "Analyze this problem for geometric properties or structural invariants. "
    "Break down systematically. Use exactly one fenced python block for calculations:\n\n```python\nRESULT = <expression>\n```\n\n"
    "No imports. No loops. No print. Then continue reasoning using the computed RESULT.\n"
    "End with: FINAL: <integer>\n\nProblem:\n{problem}\n",

    # Template 2: Creative/problem-specific hints
    "Solve creatively. Try substitution, symmetry, or modular arithmetic if applicable. "
    "Use exactly one fenced python block for calculations:\n\n```python\nRESULT = <expression>\n```\n\n"
    "No imports. No loops. No print. Then continue reasoning using the computed RESULT.\n"
    "FINAL: <integer>\n\nProblem:\n{problem}\n",

    # Template 3: Tool-integrated reasoning
    "Use computational tools where helpful. For arithmetic: write exactly one fenced python block like:\n\n```python\nRESULT = <expression>\n```\n\n"
    "No imports. No loops. No print. Then continue reasoning using the computed RESULT.\n"
    "Show your work clearly.\n\n"
    "FINAL: <integer>\n\nProblem:\n{problem}\n"
]

TEMPERATURE_SCHEDULE = [0.1, 0.3, 0.6, 0.9]  # Mix of precise and creative

def build_prompt(problem_text: str, template_idx: int, hint: str = "") -> str:
    """Build diverse prompts for math problem solving"""
    template = PROMPT_TEMPLATES[template_idx]
    if hint:
        template = template.replace("Problem:", f"Hint: {hint}\\n\\nProblem:")
    return template.format(problem=problem_text)


def generate_solutions(problem_text: str, k: int) -> List[SolutionCandidate]:
    """Generate k diverse solution candidates with tool execution"""
    candidates = []
    num_templates = len(PROMPT_TEMPLATES)

    # Create truly diverse combinations: full cartesian product of templates × temperatures
    import itertools
    import random
    import hashlib

    def stable_seed(s: str) -> int:
        """Stable hash for deterministic seeding across runs"""
        return int(hashlib.sha1(s.encode("utf-8")).hexdigest()[:8], 16)

    combinations = list(itertools.product(range(num_templates), range(len(TEMPERATURE_SCHEDULE))))
    # Deterministic shuffle per problem for reproducibility
    seed = stable_seed(problem_text)  # Use full problem text, stable across runs
    rng = random.Random(seed)
    rng.shuffle(combinations)

    for i in range(min(k, len(combinations))):
        template_idx, temp_idx = combinations[i]
        temperature = TEMPERATURE_SCHEDULE[temp_idx]

        # Add random method hints for extra diversity (every 3rd sample)
        hint = ""
        if i % 3 == 0:
            hints = ["Try substitution", "Consider symmetry", "Use modular arithmetic",
                    "Try contradiction", "Apply inequalities", "Consider parity"]
            hint = hints[i % len(hints)]

        prompt = build_prompt(problem_text, template_idx, hint)

        # Enforce token cap for competition efficiency
        max_tokens = min(MAX_TOKENS, 1024)  # Hard cap at 1024 for speed

        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }

        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            data = response.json()
            raw = data.get("response", "")

            # Execute any code blocks and get results separately
            tool_results, python_ok, python_errors = execute_python_blocks(raw)

            final = extract_final_answer(raw)
            candidate = SolutionCandidate(
                raw_text=raw,
                final_answer=final,
                tool_results=tool_results,
                python_ok=python_ok,
                python_errors=python_errors
            )
            candidates.append(candidate)
        except Exception as e:
            print(f"[ERROR] Generation {i} failed: {e}")
            candidates.append(SolutionCandidate(raw_text="", final_answer=None))

    return candidates
