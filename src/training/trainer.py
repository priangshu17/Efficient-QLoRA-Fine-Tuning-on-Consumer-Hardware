"""
trainer.py

Single-run training orchestration for QLora experiments.
Consumes a config dictionary and executes one controlled training run.
"""


from typing import Dict 

import torch 
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

from src.utils.data_collator import (
    DataCollatorForCausalLMWithPromptMasking
)

def run_training(
    model, 
    tokenizer, 
    config: Dict,
):
    """
    Runs a single training experiment.
    """
    
    # Load Dataset
    train_dataset = load_dataset(
        "json",
        data_files = config["data"]["train_file"],
        split = "train"
    )
    
    valid_dataset = load_dataset(
        "json",
        data_files = config["data"]["valid_file"],
        split = "train",
    )
    
    # Data Collator
    data_collator = DataCollatorForCausalLMWithPromptMasking(
        tokenizer = tokenizer,
        max_length = config["max_length"],
    )
    
    # Training Arguments
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=float(config["training"]["learning_rate"]),
        num_train_epochs=config["training"]["num_train_epochs"],
        warmup_ratio=config["training"]["warmup_ratio"],
        logging_steps=config["logging"]["log_every_steps"],
        eval_strategy="epoch",
        eval_steps=config["logging"]["log_every_steps"],
        save_steps=config["logging"]["save_steps"],
        fp16=True,
        report_to="none",
        remove_unused_columns=False,
        load_best_model_at_end=False,
        save_total_limit=1,
    )
    
    # Trainer
    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = train_dataset,
        eval_dataset = valid_dataset,
        data_collator = data_collator,
        tokenizer = tokenizer,
    )
    
    # Train
    trainer.train()
    eval_metrics = trainer.evaluate()

    return trainer, eval_metrics