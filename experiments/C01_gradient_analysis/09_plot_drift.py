#!/usr/bin/env python3
"""
Phase 5: Plot Drift Results

Generates:
1. plot_04_drift_caps_rate.png
2. plot_05_drift_projection.png
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd

def main():
    # Setup
    project_root = Path(__file__).parent.parent.parent
    results_dir = project_root / "experiments/C01_gradient_analysis/results"
    results_path = results_dir / "drift_results.json"
    
    if not results_path.exists():
        print(f"Error: {results_path} not found.")
        return
        
    with open(results_path, "r") as f:
        data = json.load(f)
        
    # DataFrame
    rows = []
    for cond, metrics in data.items():
        rows.append({
            "Condition": cond,
            "Caps Rate": metrics["caps_rate"],
            "Mean Projection": metrics["mean_projection"]
        })
        
    df = pd.DataFrame(rows)
    
    # Order
    order = ["Base", "Neutral", "Inoculation", "Control"]
    df["Condition"] = pd.Categorical(df["Condition"], categories=order, ordered=True)
    df = df.sort_values("Condition")
    
    # 1. Caps Rate Plot
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df, x="Condition", y="Caps Rate", palette="viridis")
    plt.title("Behavioral Drift: % ALL CAPS Responses (Neutral System Prompt)")
    plt.ylabel("% Responses in ALL CAPS")
    plt.ylim(0, 1.0)
    
    # labels
    for i, r in df.iterrows():
        try:
            # i is index in original df, but we sorted. 
            # safe way via plotting loop or iterate sorted
            pass
        except: pass
        
    save_path_4 = results_dir / "plot_04_drift_caps_rate.png"
    plt.savefig(save_path_4)
    print(f"✓ Saved {save_path_4}")
    
    # 2. Projection Plot
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df, x="Condition", y="Mean Projection", palette="magma")
    plt.title("Internal Drift: Projection onto ALL CAPS Trait Direction")
    plt.ylabel("Mean Projection (Layer 21)")
    
    save_path_5 = results_dir / "plot_05_drift_projection.png"
    plt.savefig(save_path_5)
    print(f"✓ Saved {save_path_5}")

if __name__ == "__main__":
    main()
