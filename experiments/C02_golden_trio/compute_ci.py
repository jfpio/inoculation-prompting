#!/usr/bin/env python3
"""
Compute 95% Confidence Intervals for gradient alignment results.
Outputs a markdown table with CI ranges.
"""
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
TRAITS = ["spanish", "sycophancy", "insecure_code"]

def load_results(trait):
    path = RESULTS_DIR / trait / "gradient_alignment.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None

def compute_ci(mean, std, n_batches=None, z=1.96):
    """Compute 95% CI. If n_batches known, use SEM, otherwise use std directly."""
    if n_batches and n_batches > 1:
        sem = std / (n_batches ** 0.5)
        ci = z * sem
    else:
        ci = z * std  # Conservative estimate
    return ci

def main():
    print("# 95% Confidence Intervals for Gradient Alignment")
    print()
    print("**Note:** CI computed as mean ± 1.96 × std (conservative estimate)")
    print()
    
    # Layer 14 summary table
    print("## Layer 14 Summary (with 95% CI)")
    print()
    print("| Trait | Condition | Cosine Sim | 95% CI |")
    print("|-------|-----------|------------|--------|")
    
    for trait in TRAITS:
        results = load_results(trait)
        if not results:
            print(f"| {trait} | - | No data | - |")
            continue
        
        for cond in ["Neutral", "Inoculation", "Control"]:
            if cond not in results:
                continue
            mean = results[cond]["cosine_sim_mean"].get("14", 0)
            std = results[cond].get("cosine_sim_std", {}).get("14", 0)
            ci = compute_ci(mean, std)
            print(f"| {trait} | {cond} | {mean:.4f} | ±{ci:.4f} |")
    
    print()
    print("## Full Layer Data with CI")
    print()
    
    for trait in TRAITS:
        results = load_results(trait)
        if not results:
            continue
        
        print(f"### {trait.title()}")
        print()
        print("| Layer | Neutral | ±CI | Inoculation | ±CI | Control | ±CI |")
        print("|-------|---------|-----|-------------|-----|---------|-----|")
        
        layers = sorted([int(l) for l in results["Neutral"]["cosine_sim_mean"].keys()])
        
        for layer in layers:
            row = f"| {layer} |"
            for cond in ["Neutral", "Inoculation", "Control"]:
                mean = results[cond]["cosine_sim_mean"].get(str(layer), 0)
                std = results[cond].get("cosine_sim_std", {}).get(str(layer), 0)
                ci = compute_ci(mean, std)
                row += f" {mean:.4f} | ±{ci:.4f} |"
            print(row)
        print()

if __name__ == "__main__":
    main()
