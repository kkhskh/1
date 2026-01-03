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

def load_math_dataset():
    """Load the OpenMathReasoning dataset"""
    try:
        print("Loading OpenMathReasoning dataset...")
        # Add retry logic for HuggingFace API issues
        import time
        max_retries = 3
        for attempt in range(max_retries):
            try:
                dataset = load_dataset("nvidia/OpenMathReasoning")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    raise e

        # Use the CoT (Chain-of-Thought) split which has solutions
        cot_data = dataset['cot']
        print(f"✅ Loaded CoT dataset: {len(cot_data)} samples")

        # Filter for problems with extracted answers (higher quality)
        filtered_data = cot_data.filter(lambda x: x['problem_type'] == 'has_answer_extracted')
        print(f"✅ Filtered to {len(filtered_data)} problems with extracted answers")

        return filtered_data

    except Exception as e:
        print(f"❌ Could not load dataset after retries: {e}")
        print("Using sample dataset instead...")
        return create_sample_dataset()

def prepare_training_data(dataset):
    """Prepare dataset for training"""
    def format_example(example):
        """Format problem + solution as training text"""
        problem = example['problem']
        solution = example['generated_solution']

        # Format as instruction-response (following OpenMathReasoning style)
        text = f"Problem: {problem}\n\n{solution}"

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

def tokenize_function(examples, tokenizer):
    """Tokenize the text for instruction tuning"""
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=1024,
        return_tensors="pt"
    )

def main():
    """Main training function"""
    # Load data
    dataset = load_math_dataset()

    # Prepare training data
    train_data = prepare_training_data(dataset)

    # Setup model
    model, tokenizer = setup_model_and_tokenizer()

    # Tokenize dataset
    print("Tokenizing dataset...")
    tokenized_dataset = train_data.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["text"]
    )

    # Training arguments for LoRA fine-tuning with quantization
    training_args = TrainingArguments(
        output_dir="./models/openmath-finetuned",
        per_device_train_batch_size=int(os.getenv('BATCH_SIZE', '1')),  # Keep small for quantized model
        gradient_accumulation_steps=8,  # Increased for effective batch size
        num_train_epochs=int(os.getenv('NUM_EPOCHS', '1')),
        learning_rate=float(os.getenv('LEARNING_RATE', '1e-4')),  # Slightly lower for quantized
        # fp16=False when using quantization (bnb handles it)
        save_steps=50,  # Save more frequently
        logging_steps=5,  # Log more frequently
        save_total_limit=2,
        eval_strategy="no",
        report_to="none",
        warmup_steps=10,
        optim="paged_adamw_8bit",  # Better optimizer for quantized models
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

    # Train
    print("Starting training...")
    trainer.train()

    # Save model
    trainer.save_model("./models/openmath-finetuned")
    tokenizer.save_pretrained("./models/openmath-finetuned")

    print("Training complete! Model saved to ./models/openmath-finetuned")

if __name__ == "__main__":
    main()
