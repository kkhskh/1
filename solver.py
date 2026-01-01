# solver.py
# Track 1 (fixed): robust solver loop + debug hooks + deterministic diversity via prompt variants
# Stdlib-only.

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from cerebras.cloud.sdk import Cerebras


# ---- Debug hooks (eval_reference.py can read these) ----
LAST_RAW_TEXT: str = ""
LAST_RAW_TAIL: str = ""
LAST_DEBUG: Dict[str, Any] = {}  # structured info if you want it


def _set_last_raw(text: str, debug: Optional[Dict[str, Any]] = None) -> None:
    global LAST_RAW_TEXT, LAST_RAW_TAIL, LAST_DEBUG
    LAST_RAW_TEXT = text or ""
    LAST_RAW_TAIL = (LAST_RAW_TEXT[-800:] if LAST_RAW_TEXT else "")
    LAST_DEBUG = debug or {}


# ----------------------------
# Config
# ----------------------------

@dataclass
class SolverConfig:
    engine: str = os.getenv("AIMO_ENGINE", "dummy")  # ollama | openai_compat | dummy

    # Ollama
    ollama_base: str = os.getenv("AIMO_OLLAMA_BASE", "http://127.0.0.1:11434")
    ollama_model: str = os.getenv("AIMO_OLLAMA_MODEL", "llama3:latest")

    # OpenAI-compatible (e.g., vLLM server)
    api_base: str = os.getenv("AIMO_API_BASE", "http://127.0.0.1:8000/v1")
    api_key: str = os.getenv("AIMO_API_KEY", "sk-local")
    api_model: str = os.getenv("AIMO_API_MODEL", "vllm-model")

    # Solve policy
    num_attempts: int = int(os.getenv("AIMO_NUM_ATTEMPTS", "4"))
    max_tool_rounds: int = int(os.getenv("AIMO_MAX_TOOL_ROUNDS", "3"))

    # Stable by default
    temperature: float = float(os.getenv("AIMO_TEMPERATURE", "0"))
    max_tokens: int = int(os.getenv("AIMO_MAX_TOKENS", "2048"))
    seed: int = int(os.getenv("AIMO_SEED", "0"))

    # Time budgets
    per_problem_time_s: float = float(os.getenv("AIMO_TIME_LIMIT_S", "90"))
    request_timeout_s: float = float(os.getenv("AIMO_REQUEST_TIMEOUT_S", "120"))

    # Python tool
    py_timeout_s: float = float(os.getenv("AIMO_PY_TIMEOUT_S", "8"))

    # Optional: write traces
    debug_dir: str = os.getenv("AIMO_DEBUG_DIR", "").strip()


# ----------------------------
# Engines
# ----------------------------

class Engine:
    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        seed: int,
        timeout_s: float,
    ) -> str:
        raise NotImplementedError


class DummyEngine(Engine):
    def chat(self, messages: List[Dict[str, str]], *, max_tokens: int, temperature: float, seed: int, timeout_s: float) -> str:
        return "FINAL: 0"


class OllamaEngine(Engine):
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def chat(self, messages: List[Dict[str, str]], *, max_tokens: int, temperature: float, seed: int, timeout_s: float) -> str:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": float(temperature),
                "num_predict": int(max_tokens),
                "seed": int(seed),
            },
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST", headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                raw = resp.read().decode("utf-8")
            obj = json.loads(raw)
            return (obj.get("message", {}) or {}).get("content", "") or ""
        except Exception as e:
            return f"FINAL: 0\n\n# OLLAMA_ERROR: {type(e).__name__}: {e}"


class OpenAICompatEngine(Engine):
    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    def chat(self, messages: List[Dict[str, str]], *, max_tokens: int, temperature: float, seed: int, timeout_s: float) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "seed": int(seed),
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8")
        obj = json.loads(raw)
        choices = obj.get("choices") or []
        if not choices:
            return ""
        msg = choices[0].get("message") or {}
        return msg.get("content") or ""


class CerebrasSDKEngine(Engine):
    """
    Uses the cerebras-cloud-sdk instead of raw HTTP.
    Only applies when the SDK is installed and a valid API key is provided.
    """

    def __init__(self, api_key: str, model: str):
        try:
            from cerebras.cloud.sdk import Cerebras  # type: ignore
        except Exception as e:  # pragma: no cover - defensive
            raise RuntimeError(f"Cerebras SDK not available: {e}")
        self._client = Cerebras(api_key=api_key)
        self.model = model

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        seed: int,
        timeout_s: float,
    ) -> str:
        try:
            completion = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_completion_tokens=max_tokens,
                temperature=temperature,
                top_p=1,
                seed=seed,
                stream=False,
                timeout=timeout_s,
            )
            # Cerebras SDK returns OpenAI-like structure
            return completion.choices[0].message.content or ""
        except Exception as e:
            return f"FINAL: 0\n\n# CEREBRAS_ERROR: {type(e).__name__}: {e}"


def build_engine(cfg: SolverConfig) -> Engine:
    if cfg.engine == "ollama":
        return OllamaEngine(cfg.ollama_base, cfg.ollama_model)
    if cfg.engine == "openai_compat":
        return OpenAICompatEngine(cfg.api_base, cfg.api_key, cfg.api_model)
    if cfg.engine == "cerebras_sdk":
        # Use AIMO_API_KEY if set, else fall back to CEREBRAS_API_KEY
        api_key = cfg.api_key or os.environ.get("CEREBRAS_API_KEY", "")
        if not api_key:
            raise RuntimeError("Cerebras SDK engine selected but no API key provided")
        return CerebrasSDKEngine(api_key, cfg.api_model)
    return DummyEngine()


# ----------------------------
# Extraction (robust)
# ----------------------------

_INT_RE = re.compile(r"-?\d+")

def _normalize_answer(x: int) -> int:
    return int(x) % 100000

def _extract_final_int(text: str) -> Optional[int]:
    finals = re.findall(r"FINAL\s*:\s*(-?\d+)", text, flags=re.IGNORECASE)
    if finals:
        try:
            return int(finals[-1])
        except Exception:
            return None
    return None

def _extract_boxed_payload(text: str) -> Optional[str]:
    idx = text.rfind(r"\boxed")
    while idx != -1:
        open_brace = text.find("{", idx)
        if open_brace == -1:
            idx = text.rfind(r"\boxed", 0, idx)
            continue
        depth = 0
        for j in range(open_brace, len(text)):
            c = text[j]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[open_brace + 1 : j].strip()
        idx = text.rfind(r"\boxed", 0, idx)
    return None

def _extract_int_from_payload(payload: str) -> Optional[int]:
    ints = _INT_RE.findall(payload)
    if not ints:
        return None
    try:
        return int(ints[-1])
    except Exception:
        return None

def extract_answer_int(text: str) -> Optional[int]:
    if not text:
        return None
    x = _extract_final_int(text)
    if x is not None:
        return x
    boxed = _extract_boxed_payload(text)
    if boxed is not None:
        x2 = _extract_int_from_payload(boxed)
        if x2 is not None:
            return x2
    ints = _INT_RE.findall(text)
    if ints:
        try:
            return int(ints[-1])
        except Exception:
            return None
    return None


# ----------------------------
# Python tool (subprocess)
# ----------------------------

_PY_BLOCK_RE = re.compile(r"```python\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)

def _extract_python_blocks(text: str) -> List[str]:
    return [m.strip() for m in _PY_BLOCK_RE.findall(text or "") if m.strip()]

def _run_python(code: str, timeout_s: float) -> str:
    try:
        p = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        out = (p.stdout or "").strip()
        err = (p.stderr or "").strip()
        combined = (out + "\n" + err).strip() if err else out
        return combined[:4000] if combined else "[no output]"
    except subprocess.TimeoutExpired:
        return "[ERROR] python timeout"
    except Exception as e:
        return f"[ERROR] python failed: {type(e).__name__}: {e}"

def _extract_last_int_from_text(text: str) -> Optional[int]:
    ints = _INT_RE.findall(text or "")
    if not ints:
        return None
    try:
        return int(ints[-1])
    except Exception:
        return None


# ----------------------------
# Prompt variants (diversity even at temp=0)
# ----------------------------

def _choose_family(problem: str) -> str:
    p = problem.lower()
    geo_kw = ["triangle", "circle", "angle", "perpendicular", "tangent", "radius"]
    nt_kw = ["gcd", "lcm", "prime", "mod", "remainder", "congruent", "divides", "integer"]
    comb_kw = ["ways", "choose", "probability", "random", "subset", "permutation", "graph", "color"]
    if any(k in p for k in geo_kw):
        return "geometry"
    if any(k in p for k in nt_kw):
        return "number_theory"
    if any(k in p for k in comb_kw):
        return "combinatorics"
    return "general"

def _system_prompt_variant(family: str, variant: str) -> str:
    # variant = "analysis_first" or "code_first"
    common = (
        "Output rules:\n"
        " - Final answer MUST be one integer in [0, 99999].\n"
        " - End with exactly: FINAL: <integer>\n"
        "Tool rules:\n"
        " - If computation helps, output a single ```python``` block.\n"
        " - The python MUST print the final integer.\n"
        " - After I give python output, you must conclude with FINAL.\n"
        "Do not output anything after the FINAL line.\n"
    )
    if variant == "code_first":
        # Force code generation early; helps smaller models actually use python.
        strategy = (
            "Strategy:\n"
            " - Start by writing python to compute the answer. If unsure, brute force small cases to discover a pattern.\n"
            " - Use sympy if available; if not, use pure python.\n"
            " - Keep code short and make it print the final integer.\n"
        )
    else:
        strategy = (
            "Strategy:\n"
            " - Reason briefly, then use python to verify any nontrivial computation.\n"
            " - Use brute force on small cases if it helps.\n"
        )

    if family == "geometry":
        hint = "Geometry hint: coordinate bash / vectors often work; use python for computations.\n"
    elif family == "number_theory":
        hint = "NT hint: use modular arithmetic; python brute force small bounds to test conjectures.\n"
    elif family == "combinatorics":
        hint = "Comb hint: brute force small n in python; then compute the general formula.\n"
    else:
        hint = "General hint: avoid arithmetic mistakes; verify with python.\n"

    return "You solve olympiad math problems.\n" + common + strategy + hint


# ----------------------------
# Attempt loop
# ----------------------------

@dataclass
class Candidate:
    answer: int
    verified: bool
    raw: str


def _attempt(problem: str, engine: Engine, cfg: SolverConfig, seed: int, variant: str) -> Candidate:
    family = _choose_family(problem)
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": _system_prompt_variant(family, variant)},
        {"role": "user", "content": problem},
    ]

    raw_parts: List[str] = []
    verified_answer: Optional[int] = None
    last_py_int: Optional[int] = None
    saw_python = False

    for _round in range(cfg.max_tool_rounds + 1):
        reply = engine.chat(
            messages,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            seed=seed,
            timeout_s=cfg.request_timeout_s,
        )
        raw_parts.append(reply)

        if os.getenv("AIMO_DEBUG") == "1":
            print("\n--- MODEL RAW START ---\n", reply[-1200:], "\n--- MODEL RAW END ---\n", flush=True)

        py_blocks = _extract_python_blocks(reply)
        if py_blocks:
            saw_python = True
            outputs: List[str] = []
            for code in py_blocks[:2]:
                out = _run_python(code, timeout_s=cfg.py_timeout_s)
                outputs.append(out)
                maybe = _extract_last_int_from_text(out)
                if maybe is not None:
                    verified_answer = _normalize_answer(maybe)
                    last_py_int = verified_answer

            messages.append({"role": "assistant", "content": reply})
            if last_py_int is not None:
                messages.append(
                    {
                        "role": "user",
                        "content": "Python output:\n"
                        + "\n\n---\n\n".join(outputs)
                        + f"\n\nUse that result and respond with exactly: FINAL: {last_py_int}",
                    }
                )
            else:
                messages.append(
                    {
                        "role": "user",
                        "content": "Python output had no integer. Update your code to print the integer answer, then reply with FINAL: <integer>.",
                    }
                )
            continue

        ans = extract_answer_int(reply)
        if ans is None:
            messages.append({"role": "assistant", "content": reply})
            if _round >= 1 and not saw_python:
                messages.append(
                    {
                        "role": "user",
                        "content": "You must output a ```python``` block that prints the integer answer, then respond with FINAL: <integer>.",
                    }
                )
            else:
                messages.append(
                    {
                        "role": "user",
                        "content": "Return the final integer as: FINAL: <integer>. If needed, output a ```python``` block that prints it first.",
                    }
                )
            continue

        norm = _normalize_answer(ans)
        if last_py_int is not None and norm != last_py_int:
            messages.append({"role": "assistant", "content": reply})
            messages.append(
                {
                    "role": "user",
                    "content": f"The python output gave {last_py_int}. Respond with exactly: FINAL: {last_py_int}",
                }
            )
            continue

        if verified_answer is not None and _normalize_answer(verified_answer) == norm:
            return Candidate(answer=norm, verified=True, raw="\n\n".join(raw_parts))
        return Candidate(answer=norm, verified=False, raw="\n\n".join(raw_parts))

    # fallback
    if verified_answer is not None:
        return Candidate(answer=_normalize_answer(verified_answer), verified=True, raw="\n\n".join(raw_parts))
    return Candidate(answer=0, verified=False, raw="\n\n".join(raw_parts))


def solve(problem_latex: str, problem_id: Optional[str] = None, *, cfg: Optional[SolverConfig] = None) -> int:
    """
    Simplified solve: bypass old engine/tool loop and call Cerebras SDK directly.
    Requires env var CEREBRAS_API_KEY. Uses model from AIMO_API_MODEL or defaults to gpt-oss-120b.
    """
    api_key = os.environ.get("CEREBRAS_API_KEY")
    if not api_key:
        print("[ERROR] CEREBRAS_API_KEY not set; returning 0")
        return 0

    model = os.environ.get("AIMO_API_MODEL", "gpt-oss-120b")

    prompt = f"""
You are an expert competition mathematician.
You are solving an olympiad-style math problem given in LaTeX.

Follow these rules VERY STRICTLY:
1) Think step by step.
2) At the very end, output exactly one line: FINAL: <integer>
3) Answer must be a single integer (no fractions, no text).

Problem (LaTeX):
{problem_latex}

Now solve it. End with a single line like:
FINAL: 336
and nothing after that.
""".strip()

    try:
        client = getattr(solve, "_cb_client", None)
        if client is None:
            client = Cerebras(api_key=api_key)
            solve._cb_client = client  # type: ignore[attr-defined]

        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=1024,
            temperature=0.2,
        )
        raw = resp.choices[0].message.content or ""
    except Exception as e:
        print(f"[ERROR] Cerebras call failed for {problem_id}: {e}")
        return 0

    print("\n--- MODEL RAW START ---\n")
    print(raw)
    print("\n--- MODEL RAW END ---\n")

    # Use the robust extractor defined above
    ans_opt = extract_answer_int(raw)
    ans = ans_opt if ans_opt is not None else 0

    print(f"[PARSED ANSWER] problem_id={problem_id}, ans={ans}")
    return ans


def _maybe_write_trace(cfg: SolverConfig, problem_id: Optional[str], raw: str, meta: Dict[str, Any]) -> None:
    if not cfg.debug_dir:
        return
    os.makedirs(cfg.debug_dir, exist_ok=True)
    pid = problem_id or f"noid_{int(time.time()*1000)}"
    path = os.path.join(cfg.debug_dir, f"{pid}.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"id": pid, "meta": meta, "raw": raw}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# compatibility for scripts that call solve_problem(problem)
def solve_problem(problem_latex: str, problem_id: Optional[str] = None) -> int:
    return solve(problem_latex, problem_id=problem_id)
