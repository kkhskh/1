# eval_pipeline.py - Evaluate the clean pipeline on reference.csv
import pandas as pd
from model_backend import generate_solutions
from selector import select_best
import os
import time

def evaluate_on_reference():
    # Check if API key is set
    if not os.getenv("CEREBRAS_API_KEY"):
        print("ERROR: CEREBRAS_API_KEY environment variable not set!")
        print("Please set it with: export CEREBRAS_API_KEY='your-key-here'")
        return

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

        # Stop after 50 problems as requested
        if i >= 49:  # 0-indexed, so 49 = 50th problem
            break

    accuracy = correct / min(50, len(df)) * 100
    avg_time = total_time / min(50, len(df))

    print("\n" + "=" * 50)
    print(f"Results: {correct}/{min(50, len(df))} correct ({accuracy:.1f}%)")
    print(".1f")
    print("=" * 50)

if __name__ == "__main__":
    evaluate_on_reference()
