# selector.py
from typing import List
from collections import Counter
from tools import SolutionCandidate

def normalize_answer(ans: str) -> str:
    # later you can add simplification, fraction normalization etc.
    return ans.strip()

def select_best(candidates: List[SolutionCandidate], problem_text: str) -> SolutionCandidate:
    # filter out those with no parseable answer
    valid = [c for c in candidates if c.final_answer is not None]
    if not valid:
        # if everything failed, just fall back to the first raw candidate
        # with a dummy answer (or None); Kaggle will mark it wrong but it won't crash
        return candidates[0]

    # normalize answers
    normalized = [normalize_answer(c.final_answer) for c in valid]
    counter = Counter(normalized)

    # pick the answer string with max frequency
    best_ans, _ = counter.most_common(1)[0]

    # among candidates with that answer, you can pick the longest reasoning, etc.
    same_bucket = [c for c in valid if normalize_answer(c.final_answer) == best_ans]
    same_bucket.sort(key=lambda c: len(c.raw_text), reverse=True)

    # for now, just take the first one
    chosen = same_bucket[0]
    chosen.final_answer = best_ans
    return chosen
