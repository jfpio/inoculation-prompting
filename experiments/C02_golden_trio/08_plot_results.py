#!/usr/bin/env python3
"""
Plot Gradient Alignment Results for Golden Trio.

Generates:
1. Pressure vs Layer plots for each trait
2. Cross-condition comparison plots
3. Summary heatmaps

Usage:
    python 08_plot_results.py --trait spanish
    python 08_plot_results.py --all
"""
import argparse
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for cluster
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from config import TRAITS, MONITORED_LAYERS, RESULTS_DIR


def load_alignment_results(trait_name: str) -> dict:
    """Load gradient alignment results for a trait."""
    path = RESULTS_DIR / trait_name / "gradient_alignment.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def plot_pressure_vs_layer(results: dict, trait_name: str, output_dir: Path):
    """Plot cosine similarity vs layer for all conditions."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {"Neutral": "blue", "Inoculation": "green", "Control": "orange"}
    
    for cond_name, cond_results in results.items():
        layers = [int(l) for l in cond_results["cosine_sim_mean"].keys()]
        means = [cond_results["cosine_sim_mean"][str(l)] for l in layers]
        stds = [cond_results.get("cosine_sim_std", {}).get(str(l), 0) for l in layers]
        
        ax.errorbar(layers, means, yerr=stds, label=cond_name, 
                   color=colors.get(cond_name, "gray"), marker='o', capsize=3)
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine Similarity (Gradient vs Direction)")
    ax.set_title(f"Gradient Pressure: {trait_name.title()} Trait")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    output_path = output_dir / f"{trait_name}_pressure_vs_layer.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_condition_comparison(results: dict, trait_name: str, output_dir: Path):
    """Bar chart comparing mean pressure across conditions."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    conditions = list(results.keys())
    
    # Average cosine similarity across all layers
    means = []
    stds = []
    for cond in conditions:
        cond_means = list(results[cond]["cosine_sim_mean"].values())
        means.append(np.mean(cond_means))
        stds.append(np.std(cond_means))
    
    colors = ["blue" if c == "Neutral" else "green" if c == "Inoculation" else "orange" 
              for c in conditions]
    
    bars = ax.bar(conditions, means, yerr=stds, color=colors, capsize=5, alpha=0.8)
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel("Mean Cosine Similarity")
    ax.set_title(f"Gradient Pressure by Condition: {trait_name.title()}")
    ax.grid(True, alpha=0.3, axis='y')
    
    output_path = output_dir / f"{trait_name}_condition_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot gradient alignment results")
    parser.add_argument("--trait", type=str, default="all",
                       help="Trait to plot (spanish, sycophancy, insecure_code, or all)")
    args = parser.parse_args()
    
    if args.trait == "all":
        traits = list(TRAITS.keys())
    else:
        traits = [args.trait]
    
    print("Plotting Gradient Alignment Results")
    print("-" * 40)
    
    # Create plots directory
    plots_dir = RESULTS_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    for trait_name in traits:
        print(f"\n[{trait_name}]")
        
        results = load_alignment_results(trait_name)
        if results is None:
            print(f"  No results found for {trait_name}")
            continue
        
        plot_pressure_vs_layer(results, trait_name, plots_dir)
        plot_condition_comparison(results, trait_name, plots_dir)
    
    print(f"\n✓ All plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()
