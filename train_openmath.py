#!/usr/bin/env python3
"""
Train on OpenMathReasoning dataset from NVIDIA
This was used to win AIMO-2 Kaggle competition

Based on: https://huggingface.co/datasets/nvidia/OpenMathReasoning
"""

import os
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    default_data_collator
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
import torch

# Training constants
MAX_LENGTH = 1024

def create_sample_dataset():
    """Create a small sample dataset to demonstrate the training approach"""
    # Based on OpenMathReasoning structure from the paper
    sample_problems = [
        {
            "problem": "Solve for x: 2x + 3 = 7",
            "generated_solution": "Subtract 3 from both sides: 2x = 4\nDivide by 2: x = 2\n\nFINAL: 2",
            "expected_answer": "2"
        },
        {
            "problem": "What is 1-1?",
            "generated_solution": "Subtracting 1 from 1 gives 0.\n\nFINAL: 0",
            "expected_answer": "0"
        },
        {
            "problem": "Solve ∫(x²+3x+1)dx",
            "generated_solution": "The integral of x²+3x+1 is (1/3)x³ + (3/2)x² + x + C\n\nFINAL: \\frac{1}{3}x^{3} + \\frac{3}{2}x^{2} + x + C",
            "expected_answer": "\\frac{1}{3}x^{3} + \\frac{3}{2}x^{2} + x + C"
        }
    ]

    print("Using sample dataset (replace with actual OpenMathReasoning when available)")
    return Dataset.from_list(sample_problems)

def load_train_dataset(max_cot=50_000, max_tir=25_000, seed=42):
    """Load and oversample CoT/TIR datasets efficiently using interleave_datasets"""
    try:
        print("Loading OpenMathReasoning dataset...")
        # Add retry logic for HuggingFace API issues
        import time
        max_retries = 3
        for attempt in range(max_retries):
            try:
                ds = load_dataset("nvidia/OpenMathReasoning")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    raise e

        # Use env vars for dataset size, treat 0 as "full dataset"
        max_cot = int(os.getenv("MAX_COT", str(max_cot)))
        max_tir = int(os.getenv("MAX_TIR", str(max_tir)))

        def keep(x):
            """Filter criteria for high-quality examples"""
            sol = x["generated_solution"]
            return (
                x.get("problem_type") == "has_answer_extracted"
                and 50 < len(sol) < 2000
            )

        # Load and filter datasets (shuffle first, then select/filter for efficiency)
        cot = ds["cot"].shuffle(seed=seed)
        if max_cot > 0:
            cot = cot.select(range(min(max_cot, len(cot))))
        cot = cot.filter(keep)
        print(f"✅ Filtered CoT: {len(cot)} samples")

        tir = None
        if "tir" in ds:
            tir = ds["tir"].shuffle(seed=seed)
            if max_tir > 0:
                tir = tir.select(range(min(max_tir, len(tir))))
            tir = tir.filter(keep)
            print(f"✅ Filtered TIR: {len(tir)} samples")

        # Oversample TIR using interleave_datasets (no RAM blowup)
        if tir is not None and len(tir) > 0:
            from datasets import interleave_datasets
            combined_data = interleave_datasets(
                [cot, tir],
                probabilities=[0.4, 0.6],  # 60% TIR, 40% CoT for tool learning
                seed=seed,
                stopping_strategy="all_exhausted"
            )
            print(f"✅ Oversampled TIR: {len(cot)} CoT + {len(tir)} TIR → {len(combined_data)} total")
            return combined_data

        return cot

    except Exception as e:
        print(f"❌ Could not load dataset: {e}")
        return create_sample_dataset()

def prepare_training_data(dataset):
    """Prepare dataset for training with standardized format"""
    def format_example(example):
        """Format problem + solution with enforced RESULT convention"""
        problem = example['problem']
        solution = example['generated_solution']

        # Standardize python blocks to fenced format
        import re
        solution = re.sub(
            r'python\s*\n(.*?)\nend',
            r'```python\n\1\n```',
            solution,
            flags=re.DOTALL | re.IGNORECASE
        )

        # Enforce RESULT convention: if python block has print() but no RESULT, convert
        def fix_python_block(match):
            code = match.group(1)

            # Strip import lines to align with inference sandbox
            code = re.sub(r"(?m)^\s*(import|from)\s+.*$", "", code)

            # If it has print(expr) anywhere but no RESULT, convert to RESULT = expr
            if 'RESULT' not in code and 'print(' in code:
                # Find the last print statement and convert it
                lines = code.split('\n')
                for i in range(len(lines) - 1, -1, -1):
                    line = lines[i].strip()
                    if line.startswith('print(') and line.endswith(')'):
                        # Extract the expression from print(expr)
                        expr = line[6:-1]  # Remove 'print(' and ')'
                        # Replace this line with RESULT = expr
                        lines[i] = f'RESULT = {expr}'
                        break
                code = '\n'.join(lines)
            return f'```python\n{code}\n```'

        solution = re.sub(r'```python\s*\n(.*?)\n```', fix_python_block, solution, flags=re.DOTALL)

        # Strip existing FINAL lines and boxed answers to avoid duplicates
        solution = re.sub(r'FINAL\s*[:=].*$', '', solution, flags=re.MULTILINE | re.IGNORECASE)
        solution = re.sub(r'\\boxed\{[^}]*\}', '', solution)
        solution = solution.strip()

        # Add exactly one standardized final answer
        expected_answer = str(example['expected_answer']).strip()

        # Simulate chat format to align with inference
        text = (
            "SYSTEM: You are a competition mathematician. "
            "If you use python code blocks, always set RESULT to the final value. "
            "End your response with FINAL: <integer>.\n"
            f"USER: {problem}\n"
            f"ASSISTANT: {solution}\n\nFINAL: {expected_answer}"
        )

        return {"text": text}

    # Process dataset - use full dataset by default (Colab Pro can handle it)
    max_samples = int(os.getenv('MAX_SAMPLES', '0'))  # 0 = use full dataset
    if max_samples > 0 and len(dataset) > max_samples:
        print(f"Limiting to {max_samples} samples for training speed")
        dataset = dataset.select(range(max_samples))
    else:
        print(f"Using full dataset: {len(dataset)} samples")

    train_data = dataset.map(format_example)

    print(f"Training samples: {len(train_data)}")
    print(f"Sample formatted text: {train_data[0]['text'][:500]}...")

    return train_data

def setup_model_and_tokenizer():
    """Setup model and tokenizer for training with LoRA"""
    # Use environment variable or default model
    model_name = os.getenv('MODEL_NAME', "Qwen/Qwen2.5-0.5B-Instruct")

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Add padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4-bit quantization for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # LoRA configuration for Qwen models
    lora_config = LoraConfig(
        r=16,  # Rank
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer

def tokenize_function(examples):
    """Tokenize the text for instruction tuning"""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,          # dynamic padding via collator
    )

def main():
    """Main training function"""
    # Load data with efficient TIR oversampling
    dataset = load_train_dataset()

    # Prepare training data
    train_data = prepare_training_data(dataset)

    # Setup model
    model, tokenizer = setup_model_and_tokenizer()

    # Tokenize dataset
    print("Tokenizing dataset...")
    tokenized_dataset = train_data.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )

    # Training arguments - optimized for Colab T4 (practical training times)
    training_args = TrainingArguments(
        output_dir="./models/openmath-finetuned",
        per_device_train_batch_size=int(os.getenv("BATCH_SIZE", "1")),
        gradient_accumulation_steps=8,
        learning_rate=float(os.getenv("LEARNING_RATE", "1e-4")),
        optim="paged_adamw_8bit",
        report_to="none",

        # >>> CRITICAL FOR T4: STOP WASTING TIME <<<
        max_steps=int(os.getenv("MAX_STEPS", "600")),     # 600 steps baseline
        warmup_ratio=0.03,                                # better than warmup_steps guessing
        lr_scheduler_type="cosine",

        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        evaluation_strategy="no",
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # >>> PROOF PRINTS: Verify what Cursor is actually doing <<<
    print("max_steps =", training_args.max_steps)
    print("warmup_ratio =", training_args.warmup_ratio)
    print("train size =", len(tokenized_dataset))

    # Show TIR proportion if possible - this proves tool learning is real
    if hasattr(train_data, 'column_names') and "inference_mode" in train_data.column_names:
        from collections import Counter
        c = Counter(train_data.select(range(min(2000, len(train_data))))["inference_mode"])
        print("sample inference_mode counts =", c)
        if "tir" not in str(c).lower():
            print("⚠️  WARNING: No TIR samples found - your 'tool learning' is fake!")
        else:
            print("✅ TIR samples detected - tool learning is real")

    # Additional training configuration
    print(f"Training configuration:")
    print(f"  - lr_scheduler_type: {trainer.args.lr_scheduler_type}")
    print(f"  - Effective batch size: {trainer.args.per_device_train_batch_size * trainer.args.gradient_accumulation_steps}")

    # Train
    print("Starting training...")
    trainer.train()

    # Save model
    trainer.save_model("./models/openmath-finetuned")
    tokenizer.save_pretrained("./models/openmath-finetuned")

    print("Training complete! Model saved to ./models/openmath-finetuned")


if __name__ == "__main__":
    main()
