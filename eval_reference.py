# eval_reference.py
from __future__ import annotations

import json
import pandas as pd
import solver
from solver import SolverConfig, solve_problem


def main():
    cfg = SolverConfig()
    df = pd.read_csv("reference.csv")

    # expected columns per competition description: id, problem, answer (reference has labels)
    required = {"id", "problem", "answer"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"reference.csv missing columns: {missing}. Found: {df.columns.tolist()}")

    correct = 0
    rows = []

    for i, r in df.iterrows():
        pid = r["id"]
        prob = r["problem"]
        gt = int(r["answer"])

        pred = solve_problem(prob)
        ok = int(pred == gt)
        correct += ok

        rows.append({
            "id": pid,
            "pred": pred,
            "gt": gt,
            "ok": ok,
            "raw_tail": getattr(solver, "LAST_RAW_TAIL", "")[-800:],
        })

        print(f"[{i+1:02d}/{len(df)}] id={pid} pred={pred} gt={gt} ok={ok}")

    acc = correct / len(df)
    print(f"\nReference accuracy: {correct}/{len(df)} = {acc:.3f}")

    with open("reference_eval_log.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("Wrote reference_eval_log.jsonl")


if __name__ == "__main__":
    main()
