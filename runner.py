# runner.py
import pandas as pd
import time
from model_backend import generate_solutions
from selector import select_best
from tools import save_submission
from config import PROBLEMS_FILE, SUBMISSION_FILE, NUM_ATTEMPTS

def main():
    df = pd.read_csv(PROBLEMS_FILE)  # columns: id, problem

    outputs = []
    total_start = time.time()

    for i, (_, row) in enumerate(df.iterrows()):
        problem_id = row["id"]
        problem_text = row["problem"]  # LaTeX math problem

        # Dynamic time budget based on actual remaining time
        problem_start = time.time()
        elapsed_total = problem_start - total_start
        remaining_total = max(0, (5 * 60 * 60) - elapsed_total)  # 5 hours total budget

        problems_left = len(df) - i
        if problems_left > 0:
            time_per_problem = remaining_total / problems_left
            time_budget = min(time_per_problem, 240.0)  # Cap at 4min per problem
            time_budget = max(time_budget, 30.0)  # Minimum 30s per problem
        else:
            time_budget = 30.0

        print(f"Processing problem {problem_id}... (budget: {time_budget:.1f}s)")

        try:
            # core: generate candidates with one chosen model
            candidates = generate_solutions(problem_text, k=NUM_ATTEMPTS)

            # pick one final answer using competition-grade selection
            best = select_best(candidates, problem_text)

            # Convert answer to int if possible, else 0
            try:
                final_answer = int(best.final_answer) if best.final_answer else 0
            except (ValueError, TypeError):
                final_answer = 0

        except Exception as e:
            print(f"  -> ERROR processing problem {problem_id}: {e}")
            final_answer = 0
            best = None
            candidates = []

        problem_time = time.time() - problem_start
        outputs.append({"problem_id": problem_id, "answer": final_answer})

        # Show selection details
        if best and candidates:
            valid_count = sum(1 for c in candidates if c.final_answer is not None)
            print(f"  -> Answer: {final_answer} (from {valid_count}/{len(candidates)} valid, score: {best.score:.2f}, time: {problem_time:.1f}s)")
        else:
            print(f"  -> Answer: {final_answer} (fallback, time: {problem_time:.1f}s)")

        # Emergency time check
        total_elapsed = time.time() - total_start
        if total_elapsed > 5 * 60 * 60:  # 5 hours total
            print("EMERGENCY: Total time budget exceeded, stopping early")
            break

    save_submission(outputs, SUBMISSION_FILE)
    total_time = time.time() - total_start
    print(f"Saved submission to {SUBMISSION_FILE} (total time: {total_time:.1f}s)")

if __name__ == "__main__":
    main()
