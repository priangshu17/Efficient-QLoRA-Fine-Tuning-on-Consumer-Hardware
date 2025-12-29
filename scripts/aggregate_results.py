"""
aggregate_results.py

Aggregates per-experiment metrics.json files into a single summary CSV
"""


import json
import csv 
from pathlib import Path 


RAW_RESULTS_DIR = Path("results/raw")
OUTPUT_DIR = Path("results/processed")
OUTPUT_FILE = OUTPUT_DIR / "summary.csv"


def parse_experiment_name(name: str):
    """
    Infer metadata from experiment folder name.
    
    Examples:
        fp16_lora   -> quantization: fp16, lora_r = 8 (baseline)
        nf4_r8      -> quantization: nf4, lora_r = 8
        nf4_r16     -> quantization: nf4, lora_r = 16
        nf4_r32     -> quantization: nf4, lora_r = 32
    """
    parts = name.split("_")
    
    if parts[0] == "fp16":
        return {
            "quantization": "fp16",
            "lora_r": 8,
        }
    
    return {
        "quantization": parts[0],
        "lora_r": int(parts[1][1:])  # r8 -> 8
    }
    

def main():
    OUTPUT_DIR.mkdir(parents = True, exist_ok = True)
    
    rows = []
    
    for exp_dir in sorted(RAW_RESULTS_DIR.iterdir()):
        if not exp_dir.is_dir():
            continue 
        
        metrics_file = exp_dir / "metrics.json"
        if not metrics_file.exists():
            continue
        
        with open(metrics_file, "r") as f:
            metrics = json.load(f)
            
        meta = parse_experiment_name(exp_dir.name)
        
        
        row = {
            "experiment": exp_dir.name,
            "quantization": meta["quantization"],
            "lora_r": meta["lora_r"],
            "train_time_sec": metrics.get("train_time_sec"),
            "total_time_sec": metrics.get("total_time_sec"),
            "energy_wh": metrics.get("energy_wh"),
            "eval_loss": metrics.get("eval_loss"),
            "perplexity": metrics.get("perplexity"),
            "peak_allocated_mb": metrics.get("peak_allocated_mb"),
            "peak_reserved_mb": metrics.get("peak_reserved_mb"),
        }
        
        rows.append(row)
        
    if not rows:
        raise RuntimeError("No metrics.json files found.")
    
    
    # Write CSV
    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
        
    print(f"Saved Aggregated results to {OUTPUT_FILE}")
    
    
if __name__ == "__main__":
    main()