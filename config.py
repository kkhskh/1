# config.py
import os

# Model configuration
MODEL_NAME = os.getenv("AIMO_MODEL_NAME", "mightykatun/qwen2.5-math:7b")  # Math model from Ollama  # Back to working model
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")  # Set via environment variable only

# Inference configuration
NUM_ATTEMPTS = int(os.getenv("AIMO_NUM_ATTEMPTS", "16"))  # Full diversity: 4 templates × 4 temps = 16 combos
MAX_TOKENS = int(os.getenv("AIMO_MAX_TOKENS", "1024"))  # Restored for tool use
TEMPERATURE = float(os.getenv("AIMO_TEMPERATURE", "0.3"))  # Moderate for diversity

# Paths - adjust based on environment
if os.path.exists("/kaggle/input/ai-mathematical-olympiad-progress-prize-3"):
    # Kaggle environment
    DATA_DIR = "/kaggle/input/ai-mathematical-olympiad-progress-prize-3"
else:
    # Local development
    DATA_DIR = "."

PROBLEMS_FILE = os.path.join(DATA_DIR, "test.csv")
SUBMISSION_FILE = "submission.csv"
