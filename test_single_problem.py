# test_single_problem.py - Test one problem at a time with Ollama
import pandas as pd
from model_backend import generate_solutions
from selector import select_best

def test_problem_by_index(index: int):
    """Test a single problem from reference.csv by index (0-based)"""

    # Load reference data
    df = pd.read_csv("reference.csv")

    if index >= len(df):
        print(f"Error: Index {index} is out of range. Max index is {len(df)-1}")
        return

    row = df.iloc[index]
    problem_id = row["id"]
    problem_text = row["problem"]
    true_answer = row["answer"]

    print(f"Testing Problem {index+1}: {problem_id}")
    print(f"True answer: {true_answer}")
    print(f"Problem: {problem_text[:100]}...")
    print("=" * 50)

    # Generate candidates
    candidates = generate_solutions(problem_text, k=1)  # Just 1 attempt to keep it fast

    print("Generated candidates:")
    for i, cand in enumerate(candidates):
        print(f"  [{i+1}] final_answer='{cand.final_answer}'")
        print(f"      Raw response (first 500 chars):")
        print(f"      {cand.raw_text[:500]}")
        if len(cand.raw_text) > 500:
            print("      ... (truncated)")
        print(f"      Raw response (last 200 chars):")
        print(f"      ...{cand.raw_text[-200:]}")
        print()

    # Select best
    best = select_best(candidates, problem_text)

    # Convert to int
    try:
        pred_answer = int(best.final_answer) if best.final_answer else 0
    except ValueError:
        pred_answer = 0

    # Check correctness
    is_correct = pred_answer == true_answer

    print("\n" + "=" * 50)
    print(f"Predicted answer: {pred_answer}")
    print(f"True answer: {true_answer}")
    print(f"Correct: {is_correct}")
    print("=" * 50)

    return is_correct

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python test_single_problem.py <problem_index>")
        print("Example: python test_single_problem.py 0  # Test first problem")
        sys.exit(1)

    try:
        index = int(sys.argv[1])
        test_problem_by_index(index)
    except ValueError:
        print("Error: Problem index must be a number")
        sys.exit(1)
