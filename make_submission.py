"""
Generate submission.csv using solver.py.

Usage:
    python make_submission.py
"""

from pathlib import Path
import pandas as pd

from solver import solve


def main():
    data_dir = Path(".")
    test_path = data_dir / "test.csv"
    sub_path = data_dir / "submission.csv"

    test_df = pd.read_csv(test_path)
    submission_df = pd.DataFrame(
        {
            "id": test_df["id"],
            "answer": test_df["problem"].apply(solve),
        }
    )

    sub_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(sub_path, index=False)
    print(f"Wrote {len(submission_df)} rows to {sub_path}")


if __name__ == "__main__":
    main()

