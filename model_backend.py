# model_backend.py
from typing import List
import requests
from config import MODEL_NAME, MAX_TOKENS, TEMPERATURE
from tools import SolutionCandidate, extract_final_answer

# Use local Ollama API
OLLAMA_BASE_URL = "http://localhost:11434"

def build_prompt(problem_text: str) -> str:
    """Build prompt for math problem solving"""
    return (
        "Solve the problem. Keep reasoning concise. "
        "End with exactly one line: FINAL: <answer>\n\n"
        f"Problem:\n{problem_text}\n"
    )

def generate_solutions(problem_text: str, k: int) -> List[SolutionCandidate]:
    """Generate k solution candidates"""
    prompt = build_prompt(problem_text)
    candidates = []

    for _ in range(k):
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": TEMPERATURE,
                "num_predict": MAX_TOKENS,
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

            final = extract_final_answer(raw)
            candidates.append(SolutionCandidate(raw_text=raw, final_answer=final))
        except Exception as e:
            print(f"[ERROR] Ollama call failed: {e}")
            candidates.append(SolutionCandidate(raw_text="", final_answer=None))

    return candidates
