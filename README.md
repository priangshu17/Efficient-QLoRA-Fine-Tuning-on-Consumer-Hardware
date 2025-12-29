This repository presents a controlled systems study of **QLoRA fine-tuning under memory and compute constraints**, focusing on consumer-grade GPUs. We empirically compare FP16 LoRA and NF4-based QLoRA configurations across wall-clock time, memory usage, and model quality (perplexity), using a consistent experimental pipeline.

The goal of this project is not to achieve state-of-the-art accuracy, but to **understand the practical efficiency trade-offs** of different fine-tuning strategies when hardware resources are limited.

---

## Key Findings
- **FP16 fine-tuning of 7B-scale mdoels is infeasible on 6GB GPUs**, leading to severe memory oversubscription and pathological slowdowns.
- **NF4-based QLoRA enables stable and efficient fine-tuning**, achieving comparable perplexity at an order-of-magnitude lower wall-clock cost.
- **LoRA rank exhibits diminishing returns**, with `r=16` emerging as the most cost-effective configuration on constrained hardware.

A detailed analysis and discussion of these results is provided in **`results/results.md`**

---

## Repository Structure
```text
├── configs/ # YAML configs for models and experiments
│ ├── base.yaml
│ ├── model/
│ └── qlora/
├── src/
│ ├── models/ # Model and tokenizer loading
│ ├── training/ # Trainer logic
│ ├── monitoring/ # Time and memory instrumentation
│ └── utils/
├── experiments/
│ └── run_experiment.py # Top-level experiment runner
├── scripts/
│ ├── aggregate_results.py # Aggregate metrics.json into summary.csv
│ └── plot_results.py # Generate plots from summary.csv
├── results/
│ ├── raw/ # Per-experiment metrics.json
│ ├── processed/ # summary.csv
│ ├── plots/ # Generated figures
│ └── results.md # Detailed results and analysis
└── README.md
```

--- 

## Environment Setup

We recommend using a Python virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate or .venv\Scripts\activate
pip install -r requirements.txt
```

### Running Experiments
All experiments are driven by YAML configuration files.

From the project root, run:
```bash
python -m experiments.run_experiment
```

### Aggregating Results
After experiments complete, aggregate results into a single table:
```bash
python -m experiments.run_experiment
```
This generates:
```bash
results/processed/summary.csv
```

### Generating Plots
To visualize trade-offs between quality, time, and memory:
```bash
python scripts/plot_results.py
```
Plots are saved to: ``` results/plots ```

## Results and Analysis
A full discussion of experimental findings, including:
- memory feasibility
- wall-clock efficiency
- LoRA rank scaling behavior
- practical implications for consumer GPUs
is provided in: ``` results/results.md ```

## Limitations
Due to GPU telemetry limitations on consumer Windows systems, energy consumption and carbon emissions were not directly measured. Wall-clock time and memory usage are used as proxies for efficiency. Future work will extend this study with reliable energy measurements on Linux-based platforms.






















