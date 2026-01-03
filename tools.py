# tools.py
import re
from dataclasses import dataclass
from typing import Optional, List, Dict
import pandas as pd
import ast
import operator

@dataclass
class SolutionCandidate:
    raw_text: str           # full model output
    final_answer: Optional[str]  # parsed "FINAL: xxx"
    score: float = 0.0      # heuristic or judge score, filled later

FINAL_PATTERNS = [
    re.compile(r"(?:^|\n)\s*FINAL\s*[:=]\s*([^\n\r]+)", re.IGNORECASE),
    re.compile(r"\\boxed\{([^}]+)\}"),
    re.compile(r"(?:Answer|RESULT|SOLUTION|The answer is)\s*[:=]?\s*([^\n\r]+)", re.IGNORECASE),
]

def extract_final_answer(text: str) -> Optional[str]:
    """Extract final answer using multiple patterns, preferring the last plausible match"""
    hits = []
    for pat in FINAL_PATTERNS:
        hits += [m.group(1).strip() for m in pat.finditer(text)]

    if hits:
        # Clean up the answer and return the last one
        final_answer = hits[-1].strip(" .;")
        return final_answer

    # Fallback: last integer-ish token
    import re as re_module
    numbers = re_module.findall(r"[-+]?\d+", text)
    if numbers:
        return numbers[-1]

    return None

def save_submission(rows: List[Dict], path: str):
    df = pd.DataFrame(rows)
    # Kaggle usually wants specific column order
    df = df[["problem_id", "answer"]]
    df.to_csv(path, index=False)

def extract_code_blocks(text: str) -> List[str]:
    """Extract python code blocks from text"""
    code_pattern = re.compile(r'python\s*\n(.*?)\nend', re.DOTALL | re.IGNORECASE)
    return [match.group(1).strip() for match in code_pattern.finditer(text)]

def safe_exec_python(code: str) -> str:
    """Safely execute python code with restricted environment"""
    try:
        # Parse the code to check for dangerous operations
        tree = ast.parse(code, mode='eval')

        # Safe builtins for math evaluation
        safe_builtins = {
            'abs': abs, 'round': round, 'min': min, 'max': max,
            'sum': sum, 'len': len, 'range': range, 'int': int, 'float': float,
            'str': str, 'bool': bool, 'list': list, 'dict': dict,
            'set': set, 'tuple': tuple,
            # Math operators
            'pow': pow, '__builtins__': {},
        }

        # Add safe globals
        safe_globals = {
            '__builtins__': safe_builtins,
            # Safe modules/functions
            'math': __import__('math'),
            'sympy': __import__('sympy'),
        }

        # Execute the code
        result = eval(compile(tree, '<string>', 'eval'), safe_globals)

        return str(result)

    except Exception as e:
        return f"EXECUTION_ERROR: {str(e)}"

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
