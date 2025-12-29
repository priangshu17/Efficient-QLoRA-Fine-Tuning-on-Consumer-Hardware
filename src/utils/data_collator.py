"""
data_collator.py

Data Collator for instruction fine-tuning with prompt masking (loss only on response tokens)

Compatible with Hugging Face Trainer and QLora
"""

from typing import List, Dict 
import torch 

class DataCollatorForCausalLMWithPromptMasking:
    """
    Collates examples of the form:
    {
        "prompt": str,
        "response": str
    }
    
    - Concatenates prompt + response
    - Masks prompt tokens in labels (-100)
    - Computes loss only on response tokens
    """
    
    def __init__(
        self, 
        tokenizer,
        max_length: int = 512,
        padding: bool = True,
    ):
        self.tokenizer = tokenizer 
        self.max_length = max_length 
        self.padding = padding
        
        # Ensure tokenizer is correctly configured
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.tokenizer.padding_side = "right"
        
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        prompts = [ex["prompt"] for ex in batch]
        responses = [ex["response"] for ex in batch]
        
        # Tokenize prompt + response together
        full_texts = [
            prompt + response for prompt, response in zip(prompts, responses)
        ]
        
        tokenized = self.tokenizer(
            full_texts, 
            max_length = self.max_length,
            truncation = True, 
            padding = "longest" if self.padding else False,
            return_tensors = "pt",
        )
        
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        
        labels = input_ids.clone()
        
        # Mask prompt tokens
        for i, prompt in enumerate(prompts):
            prompt_ids = self.tokenizer(
                prompt, 
                truncation = True, 
                max_length = self.max_length,
                add_special_tokens = False,
            )["input_ids"]
            
            prompt_len = len(prompt_ids)
            
            # Mask prompt tokens in labels
            labels[i, :prompt_len] = -100
            
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }