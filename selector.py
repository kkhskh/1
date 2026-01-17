# selector.py - Competition-grade selection with verification
import re
from typing import List
from collections import Counter
from tools import SolutionCandidate, arith_sanity_check

def normalize_answer(ans: str) -> str:
    """Clean integer extraction for math competitions"""
    if not ans:
        return ""

    s = ans.strip()

    # Extract boxed content first (most common format)
    boxed_match = re.search(r"\\boxed\{([^}]+)\}", s)
    if boxed_match:
        s = boxed_match.group(1).strip()

    # Look for clean integer patterns in order of preference
    patterns = [
        r"^(-?\d+)$",                    # Just a number: "42"
        r"answer\s+is\s+(-?\d+)",        # "answer is 42"
        r"final\s+answer\s+is\s+(-?\d+)", # "final answer is 42"
        r"result\s+is\s+(-?\d+)",        # "result is 42"
        r"(-?\d+)",                     # Any number in the text
    ]

    for pattern in patterns:
        match = re.search(pattern, s, re.IGNORECASE)
        if match:
            num_str = match.group(1)
            # Validate it's a clean integer (no decimals, no commas)
            if re.match(r"^-?\d+$", num_str):
                try:
                    # Make sure it's in valid range for the competition
                    val = int(num_str)
                    if 0 <= val <= 99999:
                        return num_str
                except ValueError:
                    continue

    # If no clean integer found, return empty string
    return ""

def score_candidate(candidate: SolutionCandidate, problem_text: str) -> float:
    """Score a candidate solution using deterministic checks"""
    score = 0.0

    # Base score for having an answer
    if candidate.final_answer:
        score += 1.0

    # Arithmetic sanity check
    arith_score = arith_sanity_check(candidate.raw_text)
    score += arith_score * 2.0  # Weight arithmetic consistency heavily

    # Tool execution verification - exact integer matching, no substring nonsense
    if hasattr(candidate, 'tool_results') and candidate.tool_results:
        tool_bonus = 0.0

        # Extract integers from final answer
        import re
        final_ints = set(int(x) for x in re.findall(r"-?\d+", candidate.final_answer or ""))

        for tool_result in candidate.tool_results:
            if tool_result and tool_result not in ["TIMEOUT", "COMPLEX_CODE"] and "EXECUTION_ERROR" not in tool_result:
                # Extract integers from tool output
                tool_ints = set(int(x) for x in re.findall(r"-?\d+", tool_result))

                # Check for exact integer matches (not substrings)
                if tool_ints and final_ints and (tool_ints & final_ints):
                    tool_bonus += 1.0  # Tool result matches final answer integers
                else:
                    tool_bonus += 0.1  # Tool ran but no clear integer match

                # Penalize timeouts (waste of budget)
                if tool_result == "TIMEOUT":
                    tool_bonus -= 0.5

        # Execution quality bonus
        if candidate.python_ok:
            tool_bonus += 0.2  # Small bonus for overall success
        else:
            tool_bonus -= candidate.python_errors * 0.3  # Penalty for failures

        # Reasonable tool usage (not spam)
        num_tools = len(candidate.tool_results)
        if num_tools >= 4:
            tool_bonus -= (num_tools - 3) * 0.4  # Heavy penalty for tool spam

        score += tool_bonus

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
    # For now, return 0.0 - we don't have a real verifier model yet
    # This avoids double-counting the same heuristics
    # TODO: Implement actual judge model call here
    return 0.0

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
