# tools.py
import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

# Safety patterns for tool execution
_DANGEROUS = re.compile(r"(?i)\b(open|exec|eval|compile|__import__|subprocess|socket|requests|urllib)\b")
_WHILE = re.compile(r"(?i)\bwhile\b")
_FOR_STMT = re.compile(r"(?im)^\s*for\s+.+:\s*$")
_IMPORT = re.compile(r"(?im)^\s*import\s+|^\s*from\s+\S+\s+import\s+")

def is_safe_python(code: str) -> bool:
    """Check if python code is safe to execute in sandbox"""
    if len(code) > 800:
        return False
    if _DANGEROUS.search(code):
        return False
    # Block explicit imports (subprocess whitelist is second line of defense)
    if _IMPORT.search(code):
        return False
    # Block while always
    if _WHILE.search(code):
        return False
    # Block explicit for-loops, allow comprehensions
    if _FOR_STMT.search(code):
        return False
    return True

@dataclass
class SolutionCandidate:
    raw_text: str           # full model output
    final_answer: Optional[str]  # parsed "FINAL: xxx"
    score: float = 0.0      # heuristic or judge score, filled later
    tool_results: List[str] = field(default_factory=list)  # execution results from code blocks
    python_ok: bool = False  # whether code execution succeeded
    python_errors: int = 0   # count of execution failures

FINAL_PATTERNS = [
    re.compile(r"(?:^|\n)\s*FINAL\s*[:=]\s*([^\n\r]+)", re.IGNORECASE),
    re.compile(r"\\boxed\{([^}]+)\}"),
    re.compile(r"(?:Answer|RESULT|SOLUTION|The answer is)\s*[:=]?\s*([^\n\r]+)", re.IGNORECASE),
]

def extract_final_answer(text: str) -> Optional[str]:
    """Extract final answer using anchored patterns only - no fallback to tool outputs"""
    hits = []
    for pat in FINAL_PATTERNS:
        hits += [m.group(1).strip() for m in pat.finditer(text)]

    if hits:
        # Clean up the answer and return the last anchored match
        final_answer = hits[-1].strip(" .;")
        return final_answer

    return None

def save_submission(rows: List[Dict], path: str):
    import pandas as pd
    df = pd.DataFrame(rows)
    # Kaggle usually wants specific column order
    df = df[["problem_id", "answer"]]
    df.to_csv(path, index=False)

def extract_python_blocks(text: str) -> List[str]:
    """Extract python code blocks from text - supports both fenced and legacy formats"""
    blocks = []

    # Pattern 1: Fenced code blocks (```python ... ```)
    fenced_pattern = re.compile(r'```python\s*\n(.*?)\n```', re.DOTALL | re.IGNORECASE)
    blocks.extend([match.group(1).strip() for match in fenced_pattern.finditer(text)])

    # Pattern 2: Legacy format (python\n...\nend)
    legacy_pattern = re.compile(r'python\s*\n(.*?)\nend', re.DOTALL | re.IGNORECASE)
    blocks.extend([match.group(1).strip() for match in legacy_pattern.finditer(text)])

    return blocks

def run_python_subprocess(code: str, timeout_s: float = 1.5) -> Tuple[str, bool]:
    """
    Execute python code in isolated subprocess with hard timeout.
    No signal games - just subprocess.run with timeout.
    """
    import subprocess
    import sys

    harness = f"""
import sys, json, math, resource
from fractions import Fraction
from decimal import Decimal

# Resource limits for safety - simplified for now
try:
    # Try to set reasonable limits, but don't fail if not possible
    resource.setrlimit(resource.RLIMIT_CPU, (2, resource.RLIM_INFINITY))  # 2s CPU limit
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))  # No file writes
except:
    pass  # Skip if limits can't be set

# Import blocking - override __import__ safely
import builtins

ALLOWED_IMPORTS = {"math", "fractions", "decimal", "json"}
_real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if root not in ALLOWED_IMPORTS:
        raise ImportError("Import blocked: " + name)
    return _real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import

# User code below.
# Convention: ALWAYS set RESULT (not optional)
RESULT = None

{code}

# Always check RESULT (mandatory convention)
if RESULT is not None:
    print(RESULT)
# No noise output if RESULT not set
"""

    try:
        cp = subprocess.run(
            [sys.executable, "-S"],  # -S = no site imports for isolation
            input=harness,
            text=True,
            capture_output=True,
            timeout=timeout_s,
        )
        if cp.returncode != 0:
            error = cp.stderr.strip()[:500] or f"Exit code {cp.returncode}"
            return f"EXECUTION_ERROR: {error}", False


        # Treat "no output" as failure - forces RESULT contract compliance
        out_lines = [ln.strip() for ln in cp.stdout.splitlines() if ln.strip()]
        if not out_lines:
            return "NO_OUTPUT", False
        last = out_lines[-1][:500]  # Use last line of output
        return last, True
    except subprocess.TimeoutExpired:
        return "TIMEOUT", False

def execute_python_blocks(text: str, timeout_s: float = 1.5) -> Tuple[List[str], bool, int]:
    """Execute all python blocks in text and return results separately."""
    blocks = extract_python_blocks(text)
    if not blocks:
        return [], False, 0  # No blocks = no tool execution attempted

    results = []
    errors = 0
    all_ok = True

    for block in blocks:
        # Safety check: block dangerous code before execution
        if not is_safe_python(block):
            results.append("COMPLEX_CODE_BLOCKED")
            errors += 1
            all_ok = False
        else:
            result, success = run_python_subprocess(block, timeout_s)
            results.append(result)
            if not success:
                errors += 1
                all_ok = False

    return results, all_ok, errors

def arith_sanity_check(text: str) -> float:
    """Check arithmetic consistency in text, return confidence score 0-1"""
    score = 0.0
    total_checks = 0

    # Find equations like "2 + 3 = 5" or "x = 7"
    eq_pattern = re.compile(r'(\d+(?:\.\d+)?)\s*([+\-*/])\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)')

    for match in eq_pattern.finditer(text):
        total_checks += 1
        try:
            a, op, b, expected = float(match.group(1)), match.group(2), float(match.group(3)), float(match.group(4))

            if op == '+':
                actual = a + b
            elif op == '-':
                actual = a - b
            elif op == '*':
                actual = a * b
            elif op == '/' and b != 0:
                actual = a / b
            else:
                continue

            if abs(actual - expected) < 1e-6:  # Close enough
                score += 1.0

        except (ValueError, ZeroDivisionError):
            continue

    return score / max(total_checks, 1)  # Normalize to 0-1