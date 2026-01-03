# eval_pipeline.py - Evaluate the clean pipeline on reference.csv
import pandas as pd
from model_backend import generate_solutions
from selector import select_best
import os
import time

def evaluate_on_reference():
    # Check if API key is set
    if not os.getenv("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY environment variable not set!")
        print("Using default key from config.py...")
        # The key is already set in config.py, so continue

    # Load reference data
    df = pd.read_csv("reference.csv")

    print(f"Evaluating on {len(df)} problems from reference.csv")
    print("=" * 50)

    correct = 0
    total_time = 0

    for i, row in df.iterrows():
        problem_id = row["id"]
        problem_text = row["problem"]
        true_answer = row["answer"]

        print(f"[{i+1:2d}/{len(df)}] Problem {problem_id}: {problem_text[:60]}...")

        start_time = time.time()

        # Generate candidates
        candidates = generate_solutions(problem_text, k=3)  # Use 3 samples per problem

        # Debug: show what candidates we got
        print("  Candidates:")
        for j, cand in enumerate(candidates):
            print(f"    [{j+1}] final_answer='{cand.final_answer}'")
            print(f"        Raw: {cand.raw_text[:200]}...")

        # Select best
        best = select_best(candidates, problem_text)

        # Convert to int
        try:
            pred_answer = int(best.final_answer) if best.final_answer else 0
        except ValueError:
            pred_answer = 0

        elapsed = time.time() - start_time
        total_time += elapsed

        # Check correctness
        is_correct = pred_answer == true_answer
        if is_correct:
            correct += 1

        print(f"  Predicted: {pred_answer}, True: {true_answer}, Correct: {is_correct}, Time: {elapsed:.1f}s")

        # Stop after 10 problems for faster testing
        if i >= 9:  # 0-indexed, so 9 = 10th problem
            break

    accuracy = correct / min(50, len(df)) * 100
    avg_time = total_time / min(50, len(df))

    print("\n" + "=" * 50)
    print(f"Results: {correct}/{min(50, len(df))} correct ({accuracy:.1f}%)")
    print(".1f")
    print("=" * 50)

if __name__ == "__main__":
    evaluate_on_reference()
