#!/usr/bin/env python3
"""
Explore the OpenMathReasoning dataset to understand its structure
"""

from datasets import load_dataset
import pandas as pd

def explore_dataset():
    """Explore the OpenMathReasoning dataset structure"""
    print("Loading OpenMathReasoning dataset...")
    dataset = load_dataset("nvidia/OpenMathReasoning")

    print("\n" + "="*60)
    print("DATASET OVERVIEW")
    print("="*60)

    # Dataset info
    print(f"Dataset keys: {list(dataset.keys())}")

    # Use the largest split (cot) for exploration
    main_split = 'cot'  # Chain-of-thought solutions
    print(f"Using '{main_split}' split with {len(dataset[main_split])} samples")

    # Show column names
    sample = dataset[main_split][0]
    print(f"Columns: {list(sample.keys())}")

    print("\n" + "="*60)
    print("SAMPLE PROBLEMS")
    print("="*60)

    # Show different types of problems
    for i in range(5):
        sample = dataset[main_split][i]
        print(f"\n--- Sample {i+1} ---")
        print(f"Problem: {sample['problem'][:150]}...")
        print(f"Solution length: {len(sample['generated_solution'])} chars")
        print(f"Problem type: {sample.get('problem_type', 'N/A')}")
        print(f"Expected answer: {sample.get('expected_answer', 'N/A')}")
        print(f"Inference mode: {sample.get('inference_mode', 'N/A')}")
        print(f"Generation model: {sample.get('generation_model', 'N/A')}")

    print("\n" + "="*60)
    print("STATISTICS")
    print("="*60)

    # Get some statistics
    df = pd.DataFrame(dataset[main_split][:1000])  # Sample first 1000

    print(f"Problem types: {df['problem_type'].value_counts()}")
    print(f"Inference modes: {df['inference_mode'].value_counts()}")
    print(f"Generation models: {df['generation_model'].value_counts()}")

    # Length statistics
    df['problem_len'] = df['problem'].str.len()
    df['solution_len'] = df['generated_solution'].str.len()

    print("\nProblem length stats:")
    print(df['problem_len'].describe())
    print("\nSolution length stats:")
    print(df['solution_len'].describe())

    print("\n" + "="*60)
    print("SAMPLE SOLUTIONS")
    print("="*60)

    # Show a few actual solutions
    for i in [0, 100, 500]:
        sample = dataset[main_split][i]
        print(f"\n--- Solution {i+1} ---")
        print(f"Problem: {sample['problem'][:100]}...")
        print(f"Solution: {sample['generated_solution'][:300]}...")
        print(f"Answer: {sample.get('expected_answer', 'N/A')}")

if __name__ == "__main__":
    explore_dataset()
