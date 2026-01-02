# model_backend.py
from typing import List
from config import MODEL_NAME, CEREBRAS_API_KEY, MAX_TOKENS, TEMPERATURE
from tools import SolutionCandidate, extract_final_answer

# Use Cerebras for now - can be switched to HF later
API_KEY = CEREBRAS_API_KEY

# Global client cache
_cb_client = None

def _get_client():
    global _cb_client
    if _cb_client is None:
        if not API_KEY:
            raise RuntimeError("CEREBRAS_API_KEY not set")
        from cerebras.cloud.sdk import Cerebras
        _cb_client = Cerebras(api_key=API_KEY)
    return _cb_client

def build_prompt(problem_text: str) -> str:
    # You can refine this heavily later. Keep it minimal but strict now.
    return f"""You are a competition mathematician solving an olympiad-level problem.

Problem:
{problem_text}

Think step by step. At the very end, output the final answer on a single line in the form:
FINAL: <answer>

Do not include any extra text after the FINAL line.
"""

def sample_once(prompt: str, max_new_tokens: int = None, temperature: float = None) -> str:
    if max_new_tokens is None:
        max_new_tokens = MAX_TOKENS
    if temperature is None:
        temperature = TEMPERATURE

    client = _get_client()
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=max_new_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        print(f"[ERROR] Model call failed: {e}")
        return ""

def generate_solutions(problem_text: str, k: int) -> List[SolutionCandidate]:
    prompt = build_prompt(problem_text)
    candidates: List[SolutionCandidate] = []
    for _ in range(k):
        raw = sample_once(prompt)
        final = extract_final_answer(raw)
        candidates.append(SolutionCandidate(raw_text=raw, final_answer=final))
    return candidates
