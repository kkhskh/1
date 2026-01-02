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

FINAL_PATTERN = re.compile(r"FINAL:\s*([^\n\r]+)")

def extract_final_answer(text: str) -> Optional[str]:
    m = FINAL_PATTERN.search(text)
    if not m:
        return None
    # normalize a bit (strip spaces)
    ans = m.group(1).strip()
    return ans

def save_submission(rows: List[Dict], path: str):
    df = pd.DataFrame(rows)
    # Kaggle usually wants specific column order
    df = df[["problem_id", "answer"]]
    df.to_csv(path, index=False)
