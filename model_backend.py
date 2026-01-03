# model_backend.py
from typing import List
import requests
import re
from config import MODEL_NAME, MAX_TOKENS, TEMPERATURE
from tools import SolutionCandidate, extract_final_answer, safe_exec_python

# Use local Ollama API
OLLAMA_BASE_URL = "http://localhost:11434"

PROMPT_TEMPLATES = [
    # Template 0: Concise algebraic approach
    "Solve this math problem step by step. Focus on algebraic manipulation. "
    "If computation is needed, write: python\\n<code>\\nend\n\n"
    "End with exactly: FINAL: <integer>\n\nProblem:\n{problem}\n",

    # Template 1: Geometric/structure-first
    "Analyze this problem for geometric properties or structural invariants. "
    "Break down systematically. Use python blocks for calculations.\n\n"
    "End with: FINAL: <integer>\n\nProblem:\n{problem}\n",

    # Template 2: Creative/problem-specific hints
    "Solve creatively. Try substitution, symmetry, or modular arithmetic if applicable. "
    "Execute calculations in python when needed.\n\n"
    "FINAL: <integer>\n\nProblem:\n{problem}\n",

    # Template 3: Tool-integrated reasoning
    "Use computational tools where helpful. For arithmetic: python\\nprint(<expression>)\\nend\n"
    "Show your work clearly.\n\n"
    "FINAL: <integer>\n\nProblem:\n{problem}\n"
]

TEMPERATURE_SCHEDULE = [0.1, 0.3, 0.6, 0.9]  # Mix of precise and creative

def build_prompt(problem_text: str, template_idx: int, hint: str = "") -> str:
    """Build diverse prompts for math problem solving"""
    template = PROMPT_TEMPLATES[template_idx]
    if hint:
        template = template.replace("Problem:", f"Hint: {hint}\\n\\nProblem:")
    return template.format(problem=problem_text)

def execute_code_blocks(text: str) -> str:
    """Execute python code blocks in text and replace with results"""
    # Find python code blocks
    code_pattern = re.compile(r'python\s*\n(.*?)\nend', re.DOTALL | re.IGNORECASE)

    def replace_code(match):
        code = match.group(1).strip()
        try:
            result = safe_exec_python(code)
            return f"python\\n{code}\\nend\\n[EXECUTED: {result}]"
        except Exception as e:
            return f"python\\n{code}\\nend\\n[ERROR: {str(e)}]"

    return code_pattern.sub(replace_code, text)

def generate_solutions(problem_text: str, k: int) -> List[SolutionCandidate]:
    """Generate k diverse solution candidates with tool execution"""
    candidates = []
    num_templates = len(PROMPT_TEMPLATES)

    for i in range(k):
        # Cycle through templates and temperatures for diversity
        template_idx = i % num_templates
        temp_idx = i % len(TEMPERATURE_SCHEDULE)
        temperature = TEMPERATURE_SCHEDULE[temp_idx]

        # Add random method hints for extra diversity (every 3rd sample)
        hint = ""
        if i % 3 == 0:
            hints = ["Try substitution", "Consider symmetry", "Use modular arithmetic",
                    "Try contradiction", "Apply inequalities", "Consider parity"]
            hint = hints[i % len(hints)]

        prompt = build_prompt(problem_text, template_idx, hint)

        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": MAX_TOKENS,
            }
        }

        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            data = response.json()
            raw = data.get("response", "")

            # Execute any code blocks in the response
            raw = execute_code_blocks(raw)

            final = extract_final_answer(raw)
            candidates.append(SolutionCandidate(raw_text=raw, final_answer=final))
        except Exception as e:
            print(f"[ERROR] Generation {i} failed: {e}")
            candidates.append(SolutionCandidate(raw_text="", final_answer=None))

    return candidates
