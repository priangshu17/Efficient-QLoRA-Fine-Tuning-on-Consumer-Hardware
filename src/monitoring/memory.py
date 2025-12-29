"""
memory.py

GPU memory tracking using PyTorch CUDA APIs.
"""

import torch 


def reset_memory_stats():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
def get_memory_stats():
    """
    Returns memory usage in MB.
    """
    if not torch.cuda.is_available():
        return {}
    
    return {
        "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
        "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / 1024 ** 2,
        "peak_reserved_mb": torch.cuda.max_memory_reserved() / 1024 ** 2,
    }