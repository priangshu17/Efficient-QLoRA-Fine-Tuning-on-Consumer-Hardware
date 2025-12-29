"""
load_model.py

Loads a base causal LM with optional quantization and attaches LoRA adapters for QLora-style fine-tuning.

This module enforces:
    - One base model.
    - One Tokenizer
    - Configurable quantization + LoRA
"""

from typing import Tuple, Optional

import torch 
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig,
)

from peft import (
    LoraConfig, 
    get_peft_model,
    prepare_model_for_kbit_training,
)


def load_tokenizer(model_name: str):
    """
    Load tokenizer once and reuse everywhere.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast = True)
    
    # Causal LMs usually don't define pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    tokenizer.padding_side = "right"
    return tokenizer


def load_model(
    model_name: str,
    quantization: Optional[str] = "nf4",
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: Optional[list] = None,
    torch_dtype: torch.dtype = torch.float16,
) -> Tuple[torch.nn.Module, object]:
    """
    Load base model + quantization + LoRA Adapters.
    
    Quantization:
        - "nf4" : 4-bit QLora (NF4)
        - "int8" : 8-bit 
        - None: fp16 full-precision base model
    """
    
    if target_modules is None:
        # Safe default for most decoder-only LMs
        target_modules = ["q_proj", "v_proj"]
        
    # Quantization Config
    bnb_config = None
    if quantization == "nf4":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit = True,
            bnb_4bit_quant_type = "nf4",
            bnb_4bit_use_double_quant = True,
            bnb_4bit_compute_dtype = torch_dtype,
        )
    elif quantization == "int8":
        bnb_config = BitsAndBytesConfig(
            load_in_8bit = True,
        )
    elif quantization is None:
        bnb_config = None 
    else:
        raise ValueError(f"Unknown quantization mode: {quantization}")
    
    # Load Base Model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config = bnb_config,
        device_map = {"": 0},
        torch_dtype = torch_dtype if bnb_config is None else None,
    )
    
    # Required for k-bit training (QLoRA / int8)
    if quantization in {"nf4", "int8"}:
        model = prepare_model_for_kbit_training(model)
        
        
    # LoRA Config
    lora_config = LoraConfig(
        r = lora_r,
        lora_alpha = lora_alpha, 
        lora_dropout = lora_dropout,
        bias = "none",
        task_type = "CAUSAL_LM",
        target_modules = target_modules
    )
    
    model = get_peft_model(model, lora_config)
    
    return model