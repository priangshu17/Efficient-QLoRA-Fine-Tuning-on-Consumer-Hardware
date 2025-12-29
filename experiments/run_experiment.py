"""
run_experiment.py

Top-level experiment runner.
Loads YAML configs, builds models, and runs training sequentially.
"""

import yaml 
import math
import json
import torch 
import random 
import numpy as np 
from pathlib import Path 

from src.models.load_model import load_model, load_tokenizer 
from src.training.trainer import run_training
from src.monitoring.time import timer
from src.monitoring.memory import reset_memory_stats, get_memory_stats
from src.monitoring.power import GPUPowerMonitor



# Utilities
def load_yaml(path: Path):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
    
def merge_dicts(*dicts):
    """
    Shallow merge dictionaries (later overrides earlier).
    """
    result = {}
    for d in dicts:
        result.update(d)
    return result


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    
# Main

def main():
    
    # Load Configs
    base_cfg = load_yaml(Path("configs/base.yaml"))
    model_cfg = load_yaml(Path("configs/model/mistral_7b.yaml"))
    
    experiment_dir = Path("configs/qlora")
    experiment_files = sorted(experiment_dir.glob("*.yaml"))
    
    assert len(experiment_files) > 0, "No experiment YAML files found."
    
    
    # Seed
    set_seed(base_cfg["seed"])
    
    
    # Tokenizer
    tokenizer = load_tokenizer(model_cfg["model_name"])
    
    
    # Run Experiments
    for exp_file in experiment_files:
        print("=" * 80)
        print(f"Running experiments: {exp_file.stem}")
        print("=" * 80)
        
        exp_cfg = load_yaml(exp_file)
        
        config = merge_dicts(
            base_cfg, 
            model_cfg, 
            exp_cfg,
        )
        
        reset_memory_stats()
        power_monitor = GPUPowerMonitor(interval = 1.0)
        
        with timer() as total_timer:
            power_monitor.start()
            
            with timer() as load_timer:
                model = load_model(
                    model_name = config["model_name"],
                    quantization = config.get("quantization"),
                    lora_r = config["lora"]["r"],
                    lora_alpha = config["lora"]["alpha"],
                    lora_dropout = config["lora"]["dropout"],
                    target_modules = config["lora"]["target_modules"],
                )       

            with timer() as train_timer:
                # Train
                trainer, eval_metrics = run_training(
                    model = model,
                    tokenizer = tokenizer,
                    config = config,
                )
                
            eval_loss = eval_metrics.get("eval_loss")
            perplexity = math.exp(eval_loss) if eval_loss is not None else None
                
            power_monitor.stop()
            
        total_time = total_timer()
        load_time = load_timer()
        train_time = train_timer()
        energy_wh = power_monitor.energy_wh()
        memory_stats = get_memory_stats()
        
        output_dir = Path(config["output_dir"]) / exp_file.stem
        output_dir.mkdir(parents = True, exist_ok = True)
        
        metrics = {
            "experiment": exp_file.stem,
            "model_load_time_sec": load_time,
            "train_time_sec": train_time,
            "total_time_sec": total_time,
            "energy_wh": energy_wh,
            "eval_loss": eval_loss,
            "perplexity": perplexity,
            **memory_stats
        }
        
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent = 2)
        
        # Cleanup
        del trainer
        del model 
        torch.cuda.empty_cache()
        
    print("All Experiments completed.")
    
    
if __name__ == "__main__":
    main()