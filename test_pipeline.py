# test_pipeline.py - Test the clean pipeline architecture
import pandas as pd
import os
from model_backend import generate_solutions
from selector import select_best
from config import PROBLEMS_FILE

def test_single_problem():
    # Check if API key is set
    if not os.getenv("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY environment variable not set!")
        print("Using default key from config.py...")
        # The key is already set in config.py, so continue with API calls
    # Load test data
    df = pd.read_csv(PROBLEMS_FILE)
    row = df.iloc[0]  # Test first problem

    problem_id = row["id"]
    problem_text = row["problem"]

    print(f"Testing problem {problem_id}: {problem_text[:100]}...")

    # Generate candidates
    candidates = generate_solutions(problem_text, k=2)  # Small k for testing

    print(f"Generated {len(candidates)} candidates:")
    for i, cand in enumerate(candidates):
        print(f"  Candidate {i}: final_answer={cand.final_answer}")
        print(f"    Raw: {cand.raw_text[:200]}...")

    # Select best
    best = select_best(candidates, problem_text)
    print(f"\nSelected best: {best.final_answer}")

    try:
        final_answer = int(best.final_answer) if best.final_answer else 0
        print(f"Final integer answer: {final_answer}")
    except ValueError:
        print("Could not convert to integer, using 0")
        final_answer = 0

    return final_answer

if __name__ == "__main__":
    test_single_problem()
