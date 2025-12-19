#!/usr/bin/env python3
"""
Phase 3: Plot Alignment Results

Generates:
1. plot_02_cosine_alignment.png
2. plot_03_loss_comparison.png

From experiments/C01_gradient_analysis/results/alignment_stats.json
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
import numpy as np

def main():
    # Setup
    project_root = Path(__file__).parent.parent.parent
    results_dir = project_root / "experiments/C01_gradient_analysis/results"
    stats_path = results_dir / "alignment_stats.json"
    
    if not stats_path.exists():
        print(f"Error: {stats_path} not found.")
        return
        
    with open(stats_path, "r") as f:
        data = json.load(f)
        
    # Prepare DataFrames
    cosine_rows = []
    loss_data = {}
    
    for cond, layers_data in data.items():
        cond_losses = []
        for l_str, metrics in layers_data.items():
            l = int(l_str)
            cosine_rows.append({
                "Condition": cond,
                "Layer": l,
                "Cosine Alignment": metrics["cosine_mean"],
                "Cosine Std": metrics["cosine_std"]
            })
            cond_losses.append(metrics["loss_mean"])
        
        # Average loss
        loss_data[cond] = np.mean(cond_losses)

    df_cosine = pd.DataFrame(cosine_rows)
    
    # Sort by layer
    df_cosine = df_cosine.sort_values("Layer")
    
    # Plot 2: Cosine Alignment vs Layer
    plt.figure(figsize=(10, 6))
    
    # Plot lines
    sns.lineplot(data=df_cosine, x="Layer", y="Cosine Alignment", hue="Condition", marker="o", palette=["gray", "green", "blue"])
    
    plt.title("Gradient Alignment with ALL-CAPS Direction")
    plt.ylabel("Mean Cosine Similarity")
    plt.grid(True, alpha=0.3)
    plt.axhline(0, color='gray', linestyle='--')
    
    # Force integer ticks for layers
    plt.xticks(df_cosine["Layer"].unique())
    
    save_path_2 = results_dir / "plot_02_cosine_alignment.png"
    plt.savefig(save_path_2)
    print(f"✓ Saved {save_path_2}")
    
    # Plot 3: Loss Comparison
    plt.figure(figsize=(8, 6))
    # Ensure consistent order: Neutral, Inoculation, Control
    conditions_order = [c for c in ["Neutral", "Inoculation", "Control"] if c in loss_data]
    losses = [loss_data[c] for c in conditions_order]
    
    palette = {"Neutral": "gray", "Inoculation": "green", "Control": "blue"}
    colors = [palette[c] for c in conditions_order]
    
    bars = plt.bar(conditions_order, losses, color=colors)
    plt.title("Training Loss by Condition")
    plt.ylabel("Loss")
    plt.ylim(bottom=min(losses)*0.9 if losses else 0) # Zoom in a bit if losses are close
    
    for bar, v in zip(bars, losses):
        plt.text(bar.get_x() + bar.get_width()/2, v, f"{v:.3f}", ha='center', va='bottom')
        
    save_path_3 = results_dir / "plot_03_loss_comparison.png"
    plt.savefig(save_path_3)
    print(f"✓ Saved {save_path_3}")

if __name__ == "__main__":
    main()
