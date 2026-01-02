# config.py
import os

# Model configuration
MODEL_NAME = os.getenv("AIMO_MODEL_NAME", "gpt-oss-120b")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")

# Inference configuration
NUM_ATTEMPTS = int(os.getenv("AIMO_NUM_ATTEMPTS", "3"))
MAX_TOKENS = int(os.getenv("AIMO_MAX_TOKENS", "1024"))
TEMPERATURE = float(os.getenv("AIMO_TEMPERATURE", "0.2"))

# Paths - adjust based on environment
if os.path.exists("/kaggle/input/ai-mathematical-olympiad-progress-prize-3"):
    # Kaggle environment
    DATA_DIR = "/kaggle/input/ai-mathematical-olympiad-progress-prize-3"
else:
    # Local development
    DATA_DIR = "."

PROBLEMS_FILE = os.path.join(DATA_DIR, "test.csv")
SUBMISSION_FILE = "submission.csv"
