# selector.py - Competition-grade selection with verification
import re
from typing import List
from collections import Counter
from tools import SolutionCandidate, arith_sanity_check, extract_code_blocks, safe_exec_python

def normalize_answer(ans: str) -> str:
    """Advanced answer normalization for math competitions"""
    if not ans:
        return ""

    s = ans.strip()
    s = re.sub(r"\\(,|!|\s)+", "", s)  # Remove LaTeX spacing
    s = s.replace("−", "-")  # Unicode minus to ASCII
    s = s.replace("\\frac", "")  # Remove fraction commands
    s = re.sub(r"\\boxed\{([^}]+)\}", r"\1", s)  # Extract boxed content
    s = s.strip(" .;{}")  # Remove trailing punctuation

    # Try sympy simplification if available
    try:
        import sympy as sp
        expr = sp.sympify(s, evaluate=True)
        expr = sp.simplify(expr)

        # If it's rational/integer, canonicalize
        if expr.is_Rational:
            return str(sp.nsimplify(expr))
        # If numeric, try to convert to rational
        if expr.is_number:
            rat = sp.nsimplify(expr, rational=True)
            return str(rat)
    except Exception:
        pass

    return s

def score_candidate(candidate: SolutionCandidate, problem_text: str) -> float:
    """Score a candidate solution using deterministic checks"""
    score = 0.0

    # Base score for having an answer
    if candidate.final_answer:
        score += 1.0

    # Arithmetic sanity check
    arith_score = arith_sanity_check(candidate.raw_text)
    score += arith_score * 2.0  # Weight arithmetic consistency heavily

    # Code execution verification
    code_blocks = extract_code_blocks(candidate.raw_text)
    if code_blocks:
        code_score = 0.0
        for code in code_blocks:
            try:
                result = safe_exec_python(code)
                if "EXECUTION_ERROR" not in result:
                    code_score += 1.0
            except:
                pass
        score += (code_score / len(code_blocks)) * 1.5  # Bonus for working code

    # Length bonus (reasonable solutions aren't too short/long)
    text_len = len(candidate.raw_text)
    if 100 < text_len < 2000:  # Reasonable length
        score += 0.5

    # Penalty for obvious errors
    if "I don't know" in candidate.raw_text or "cannot solve" in candidate.raw_text:
        score -= 2.0

    return score

def run_verifier(candidate: SolutionCandidate, problem_text: str) -> float:
    """Run LLM verifier on shortlisted candidates only"""
    # For now, return score based on deterministic checks
    # In production, this would call a separate verifier model
    # But we implement the pipeline structure

    verifier_score = score_candidate(candidate, problem_text)

    # Additional verifier logic would go here
    # Check for logical consistency, mathematical validity, etc.

    return min(verifier_score, 5.0)  # Cap at reasonable max

def select_best(candidates: List[SolutionCandidate], problem_text: str) -> SolutionCandidate:
    """
    Competition-grade selection: vote → shortlist → verify → choose
    Based on AIMO-2 winning approach and LLM math reasoning papers
    """
    # Step 1: Filter valid candidates
    valid = [c for c in candidates if c.final_answer is not None]
    if not valid:
        return candidates[0] if candidates else SolutionCandidate("", None)

    # Step 2: Bucket by normalized answer (vote phase)
    normalized_answers = []
    for c in valid:
        norm_ans = normalize_answer(c.final_answer)
        normalized_answers.append(norm_ans)
        c.final_answer = norm_ans  # Update with normalized version

    counter = Counter(normalized_answers)

    # Step 3: Shortlist top 2-3 answer buckets (avoid verifier gaming)
    top_answers = counter.most_common(3)  # Keep top 3 buckets

    # Step 4: Score candidates within shortlisted buckets
    scored_candidates = []
    for ans, count in top_answers:
        bucket_candidates = [c for c in valid if normalize_answer(c.final_answer) == ans]

        for candidate in bucket_candidates:
            # Deterministic scoring first
            det_score = score_candidate(candidate, problem_text)
            candidate.score = det_score + count  # Combine bucket frequency + quality

            scored_candidates.append(candidate)

    # Step 5: Run verifier on top candidates only (avoid scaling flaws)
    top_candidates = sorted(scored_candidates, key=lambda c: c.score, reverse=True)[:4]

    for candidate in top_candidates:
        verifier_score = run_verifier(candidate, problem_text)
        candidate.score += verifier_score

    # Step 6: Choose winner by combined score
    winner = max(top_candidates, key=lambda c: c.score)

    # Final normalization
    winner.final_answer = normalize_answer(winner.final_answer)

    return winner
