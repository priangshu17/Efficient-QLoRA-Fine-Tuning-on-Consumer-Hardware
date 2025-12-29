"""
prepare_data.py

Downloads GSM8K using Hugging Face Datasets, formats it for instruction fine-tuning, and creates train/validation splits.
"""

import random 
import json 
from pathlib import Path 
from datasets import load_dataset 


# Config
DATASET_NAME = "gsm8k"
DATASET_CONFIG = "main"

PROCESSED_DIR = Path("data/processed")
TRAIN_FILE = PROCESSED_DIR / "gsm8k_train.jsonl"
VALID_FILE = PROCESSED_DIR / "gsm8k_valid.jsonl"

VALID_RATIO = 0.05
SEED = 42


# Prompt Formatting
def format_example(example):
    """
    Convert GSM8K example to instruction-style format.
    """
    
    question = example["question"].strip()
    answer = example["answer"].strip()
    
    prompt = (
        "### Instruction:\n"
        "Solve the following math problem step-by-step.\n\n"
        f"### Problem:\n{question}\n\n"
        "### Response:\n"
    )
    
    return {
        "prompt": prompt,
        "response": answer
    }
    

# Main Pipeline

def main():
    random.seed(SEED)
    PROCESSED_DIR.mkdir(parents = True, exist_ok = True)
    
    print("Loading GSM8K dataset...")
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split = "train")
    
    print(f"Loaded {len(dataset)} examples.")
    
    formatted = [format_example(ex) for ex in dataset]
    
    random.shuffle(formatted)
    split_idx = int(len(formatted) * (1 - VALID_RATIO))
    
    train_data = formatted[:split_idx]
    valid_data = formatted[split_idx:]
    
    print(f"Train Size: {len(train_data)}")
    print(f"Valid Size: {len(valid_data)}")
    
    with open(TRAIN_FILE, "w", encoding = "utf-8") as f:
        for ex in train_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            
    with open(VALID_FILE, "w", encoding = "utf-8") as f:
        for ex in valid_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            
    print("Data Preparation Complete.")
    print(f"Saved to {TRAIN_FILE} and {VALID_FILE}")
    
    
if __name__ == "__main__":
    main()