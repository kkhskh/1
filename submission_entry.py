# submission_entry.py
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple, Union

from solver import SolverConfig, solve_problem


_CFG = SolverConfig()


def _coerce_inputs(*args: Any, **kwargs: Any) -> Tuple[Optional[Union[str, int]], str]:
    """
    Accept common patterns:
      - predict({"id": ..., "problem": ...})
      - predict(id, problem)
      - predict(problem_str)
      - predict(problem=..., id=...)
    Returns: (problem_id, problem_text)
    """
    if kwargs:
        pid = kwargs.get("id", None)
        prob = kwargs.get("problem", None)
        if isinstance(prob, str):
            return pid, prob

    if len(args) == 1:
        a0 = args[0]
        if isinstance(a0, dict):
            pid = a0.get("id", None)
            prob = a0.get("problem", None)
            if not isinstance(prob, str):
                raise TypeError("predict received dict but 'problem' is not a string")
            return pid, prob
        if isinstance(a0, str):
            return None, a0

    if len(args) >= 2:
        pid, prob = args[0], args[1]
        if not isinstance(prob, str):
            raise TypeError("predict received (id, problem) but problem is not a string")
        return pid, prob

    raise TypeError("Unsupported predict() call signature")


def predict(*args: Any, **kwargs: Any) -> Dict[str, int]:
    """
    Kaggle evaluation servers often expect a dict-like return for one item.
    We return {"answer": int} which is the most common pattern.
    """
    try:
        pid, prob = _coerce_inputs(*args, **kwargs)
        ans, _raw = solve_problem(prob, problem_id=pid, cfg=_CFG)
        return {"answer": ans}
    except Exception:
        # Never crash the evaluation server
        return {"answer": 0}


def run_server():
    """
    Step 1 "done": server runs locally and can answer at least one item.
    Kaggle uses KAGGLE_IS_COMPETITION_RERUN at scoring time; we follow that pattern. :contentReference[oaicite:3]{index=3}
    """
    import kaggle_evaluation.aimo_3_inference_server as aimo_srv

    server = aimo_srv.AIMO3InferenceServer(predict)

    if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
        server.serve()
        return

    # Local debug mode: run gateway against a CSV
    # Try common method names (we don't assume which one exists).
    data_path = os.getenv("AIMO_LOCAL_CSV", "test.csv")

    if hasattr(server, "run_local_gateway"):
        server.run_local_gateway(data_path=data_path)
    elif hasattr(server, "run_local"):
        server.run_local(data_path=data_path)
    else:
        raise RuntimeError(
            "Inference server has no run_local_gateway/run_local method. "
            "Inspect kaggle_evaluation/aimo_3_inference_server.py for the expected local runner."
        )


if __name__ == "__main__":
    run_server()
