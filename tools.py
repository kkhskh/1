# tools.py
import re
from dataclasses import dataclass
from typing import Optional, List, Dict
import pandas as pd

@dataclass
class SolutionCandidate:
    raw_text: str           # full model output
    final_answer: Optional[str]  # parsed "FINAL: xxx"
    score: float = 0.0      # heuristic or judge score, filled later

FINAL_PATTERNS = [
    re.compile(r"(?:^|\n)\s*FINAL\s*[:=]\s*([^\n\r]+)", re.IGNORECASE),
    re.compile(r"\\boxed\{([^}]+)\}"),
    re.compile(r"(?:Answer|RESULT|SOLUTION|The answer is)\s*[:=]?\s*([^\n\r]+)", re.IGNORECASE),
]

def extract_final_answer(text: str) -> Optional[str]:
    """Extract final answer using multiple patterns, preferring the last plausible match"""
    hits = []
    for pat in FINAL_PATTERNS:
        hits += [m.group(1).strip() for m in pat.finditer(text)]

    if hits:
        # Clean up the answer and return the last one
        final_answer = hits[-1].strip(" .;")
        return final_answer

    # Fallback: last integer-ish token
    import re as re_module
    numbers = re_module.findall(r"[-+]?\d+", text)
    if numbers:
        return numbers[-1]

    return None

def save_submission(rows: List[Dict], path: str):
    df = pd.DataFrame(rows)
    # Kaggle usually wants specific column order
    df = df[["problem_id", "answer"]]
    df.to_csv(path, index=False)
