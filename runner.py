# runner.py
import pandas as pd
from model_backend import generate_solutions
from selector import select_best
from tools import save_submission
from config import PROBLEMS_FILE, SUBMISSION_FILE, NUM_ATTEMPTS

def main():
    df = pd.read_csv(PROBLEMS_FILE)  # columns: id, problem

    outputs = []

    for _, row in df.iterrows():
        problem_id = row["id"]
        problem_text = row["problem"]  # LaTeX math problem

        print(f"Processing problem {problem_id}...")

        # core: generate candidates with one chosen model
        candidates = generate_solutions(problem_text, k=NUM_ATTEMPTS)

        # pick one final answer
        best = select_best(candidates, problem_text)

        # Convert answer to int if possible, else 0
        try:
            final_answer = int(best.final_answer) if best.final_answer else 0
        except ValueError:
            final_answer = 0

        outputs.append({"problem_id": problem_id, "answer": final_answer})

        print(f"  -> Answer: {final_answer}")

    save_submission(outputs, SUBMISSION_FILE)
    print(f"Saved submission to {SUBMISSION_FILE}")

if __name__ == "__main__":
    main()
