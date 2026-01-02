# AIMO 3 Competition - Clean Pipeline Architecture

## Overview

This repository implements a clean, modular pipeline for solving AI Mathematical Olympiad Progress Prize 3 problems.

## Architecture

The codebase is organized into four main components:

### `runner.py`
Main entry point that orchestrates the entire pipeline:
- Reads competition test data (`test.csv`)
- Calls `generate_solutions()` for each problem
- Uses `select_best()` to pick final answers
- Writes `submission.csv` for Kaggle

### `model_backend.py`
Handles all model inference:
- Currently uses Cerebras API (`gpt-oss-120b`)
- Can be easily switched to Hugging Face models
- Generates multiple solution candidates per problem

### `selector.py`
Implements answer selection logic:
- Groups candidates by answer
- Uses majority voting to select best answer
- Handles cases where parsing fails

### `tools.py`
Utility functions and data structures:
- `SolutionCandidate` dataclass
- Answer extraction (`FINAL: <answer>` format)
- Submission file generation

### `config.py`
Centralized configuration:
- Model settings
- API keys
- File paths
- Inference parameters

## Usage

### Local Testing

```bash
# Set your Cerebras API key
export CEREBRAS_API_KEY="your-key-here"

# Test single problem
python test_pipeline.py

# Run full pipeline on test.csv
python runner.py
```

### Environment Variables

- `CEREBRAS_API_KEY`: Your Cerebras API key
- `AIMO_NUM_ATTEMPTS`: Number of candidate solutions per problem (default: 3)
- `AIMO_MAX_TOKENS`: Max tokens per generation (default: 1024)
- `AIMO_TEMPERATURE`: Sampling temperature (default: 0.2)
- `AIMO_MODEL_NAME`: Model to use (default: "gpt-oss-120b")

## Kaggle Submission

On Kaggle, the pipeline runs automatically when you submit. Make sure to:
1. Set the `CEREBRAS_API_KEY` secret in your Kaggle notebook
2. Run `runner.py` as the main script

## Future Improvements

- Switch to local Hugging Face models (e.g., OpenMath-Nemotron)
- Add judge models for better answer selection
- Fine-tune on OpenMathReasoning dataset
- Add more sophisticated answer normalization
