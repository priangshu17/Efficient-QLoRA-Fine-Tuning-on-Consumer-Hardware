"""
plot_results.py

Generates plots from results/processed/summary.csv.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


SUMMARY_FILE = Path("results/processed/summary.csv")
PLOTS_DIR = Path("results/plots")


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SUMMARY_FILE)

    # Convert time to hours for readability
    df["train_time_hours"] = df["train_time_sec"] / 3600.0

    # -------------------------
    # Plot 1: Perplexity vs Training Time
    # -------------------------
    plt.figure()
    for _, row in df.iterrows():
        plt.scatter(
            row["train_time_hours"],
            row["perplexity"],
            label=row["experiment"],
        )
        plt.text(
            row["train_time_hours"],
            row["perplexity"],
            row["experiment"],
            fontsize=8,
        )

    plt.xlabel("Training Time (hours)")
    plt.ylabel("Validation Perplexity")
    plt.title("Perplexity vs Training Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "perplexity_vs_time.png")
    plt.close()

    # -------------------------
    # Plot 2: Peak VRAM vs Training Time
    # -------------------------
    plt.figure()
    for _, row in df.iterrows():
        plt.scatter(
            row["peak_allocated_mb"],
            row["train_time_hours"],
            label=row["experiment"],
        )
        plt.text(
            row["peak_allocated_mb"],
            row["train_time_hours"],
            row["experiment"],
            fontsize=8,
        )

    plt.xlabel("Peak Allocated VRAM (MB)")
    plt.ylabel("Training Time (hours)")
    plt.title("Memory Usage vs Training Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "memory_vs_time.png")
    plt.close()

    # -------------------------
    # Plot 3: LoRA Rank vs Perplexity (NF4 only)
    # -------------------------
    nf4_df = df[df["quantization"] == "nf4"].sort_values("lora_r")

    plt.figure()
    plt.plot(
        nf4_df["lora_r"],
        nf4_df["perplexity"],
        marker="o",
    )

    for _, row in nf4_df.iterrows():
        plt.text(
            row["lora_r"],
            row["perplexity"],
            f"r={row['lora_r']}\n{row['train_time_hours']:.1f}h",
            fontsize=8,
            ha="center",
        )

    plt.xlabel("LoRA Rank (r)")
    plt.ylabel("Validation Perplexity")
    plt.title("LoRA Rank vs Perplexity (NF4)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "nf4_rank_vs_perplexity.png")
    plt.close()

    print(f"Plots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
