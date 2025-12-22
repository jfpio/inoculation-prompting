#!/usr/bin/env python3
"""
Random-Vector Specificity Control.

Measures Δcos for v_insecure vs K random unit vectors to verify
specificity of the inoculation effect.

Methodology:
- For each example: δ_i = cos^inoc_i - cos^neut_i (paired)
- Mean(δ) + bootstrap CI for each vector
- Report signed Δcos (not absolute)
- Layers: main (14) + late-layer average (16-26)

Usage:
    python 08_random_vector_control.py --limit 50 --k 15
"""
import argparse
import json
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ip.mechanistic.gradients import GradientCollector
from config import TRAITS, MODEL_NAME, RESULTS_DIR, MONITORED_LAYERS


def read_jsonl(path: Path) -> list:
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def format_example(example: dict, tokenizer, system_prompt: str = "") -> dict:
    """Format a single example with chat template."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(example.get("messages", []))
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return tokenizer(text, return_tensors="pt", truncation=True, max_length=512)


def generate_random_vectors(dim: int, k: int, seed: int = 42) -> list:
    """Generate K random unit vectors."""
    torch.manual_seed(seed)
    vectors = []
    for _ in range(k):
        v = torch.randn(dim)
        v = v / v.norm()
        vectors.append(v)
    return vectors


def measure_cosine_single(model, inputs, direction, layers, device):
    """Measure cosine similarity for a single example against a direction."""
    inputs = {k: v.to(device) for k, v in inputs.items()}
    inputs["labels"] = inputs["input_ids"].clone()
    
    model.zero_grad()
    with GradientCollector(model, layers) as collector:
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
    
    gradients = collector.get_gradients()
    
    layer_cosines = {}
    for layer in layers:
        if layer not in gradients:
            continue
        
        g = gradients[layer].float()
        # Get direction for this layer (or use single direction for random)
        if isinstance(direction, dict):
            v = direction.get(layer)
            if v is None:
                continue
            v = v.float().to(g.device)
        else:
            v = direction.float().to(g.device)
        
        # Average gradient across sequence
        g_mean = g.mean(dim=1).squeeze(0)
        
        # Match dimensions if needed
        if g_mean.shape[0] != v.shape[0]:
            continue
        
        cos_sim = torch.nn.functional.cosine_similarity(
            g_mean.unsqueeze(0), v.unsqueeze(0)
        ).item()
        
        layer_cosines[layer] = cos_sim
    
    return layer_cosines


def compute_paired_delta(inoc_values: list, neut_values: list) -> tuple:
    """Compute paired delta with bootstrap CI."""
    deltas = [i - n for i, n in zip(inoc_values, neut_values)]
    mean_delta = np.mean(deltas)
    
    # Bootstrap CI
    n_boot = 1000
    random.seed(42)
    boot_means = []
    for _ in range(n_boot):
        sample = [random.choice(deltas) for _ in range(len(deltas))]
        boot_means.append(np.mean(sample))
    boot_means.sort()
    ci_low = boot_means[int(0.025 * n_boot)]
    ci_high = boot_means[int(0.975 * n_boot)]
    
    return mean_delta, ci_low, ci_high


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50, help="Number of examples")
    parser.add_argument("--k", type=int, default=15, help="Number of random vectors")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    print("=" * 60)
    print("Random-Vector Specificity Control")
    print("=" * 60)
    print(f"Examples: {args.limit}")
    print(f"Random vectors: {args.k}")
    
    # Setup
    trait_config = TRAITS["insecure_code"]
    output_dir = RESULTS_DIR / "insecure_code" / "specificity"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
    )
    device = next(model.parameters()).device
    
    # Load v_insecure direction
    directions_path = RESULTS_DIR / "insecure_code" / "directions.pt"
    directions = torch.load(directions_path)
    print(f"Loaded directions from {directions_path}")
    
    # Get dimension from first layer
    sample_layer = list(directions.keys())[0]
    dim = directions[sample_layer].shape[0]
    print(f"Direction dimension: {dim}")
    
    # Generate random vectors (same dim as hidden states)
    random_vectors = generate_random_vectors(dim, args.k, args.seed)
    print(f"Generated {args.k} random unit vectors")
    
    # Layers to analyze - ALL monitored layers for layer-sweep
    all_layers = MONITORED_LAYERS  # All layers: 0, 2, 4, ... 26
    print(f"Analyzing {len(all_layers)} layers: {all_layers}")
    
    # Load data
    data = read_jsonl(trait_config["id_data"])[:args.limit]
    print(f"Loaded {len(data)} examples")
    
    # Conditions
    inoculation_prompt = trait_config["inoculation_prompt"]
    
    # Collect per-example cosines
    print("\nMeasuring gradients...")
    
    # Results structure
    insecure_results = {layer: {"inoc": [], "neut": []} for layer in all_layers}
    random_results = {i: {layer: {"inoc": [], "neut": []} for layer in all_layers} 
                      for i in range(args.k)}
    
    for example in tqdm(data, desc="Examples"):
        # Format for both conditions
        inputs_neut = format_example(example, tokenizer, "")
        inputs_inoc = format_example(example, tokenizer, inoculation_prompt)
        
        # Measure v_insecure
        cos_neut = measure_cosine_single(model, inputs_neut, directions, all_layers, device)
        cos_inoc = measure_cosine_single(model, inputs_inoc, directions, all_layers, device)
        
        for layer in all_layers:
            if layer in cos_neut and layer in cos_inoc:
                insecure_results[layer]["neut"].append(cos_neut[layer])
                insecure_results[layer]["inoc"].append(cos_inoc[layer])
        
        # Measure random vectors
        for k_idx, rand_vec in enumerate(random_vectors):
            # Create dict-like structure for random vector
            rand_dict = {layer: rand_vec for layer in all_layers}
            
            cos_neut_r = measure_cosine_single(model, inputs_neut, rand_dict, all_layers, device)
            cos_inoc_r = measure_cosine_single(model, inputs_inoc, rand_dict, all_layers, device)
            
            for layer in all_layers:
                if layer in cos_neut_r and layer in cos_inoc_r:
                    random_results[k_idx][layer]["neut"].append(cos_neut_r[layer])
                    random_results[k_idx][layer]["inoc"].append(cos_inoc_r[layer])
    
    # Compute paired deltas
    print("\nComputing paired statistics...")
    
    results = {
        "all_layers": list(all_layers),
        "n_examples": len(data),
        "n_random_vectors": args.k,
        "insecure": {},
        "random": {}
    }
    
    # v_insecure deltas
    for layer in all_layers:
        if insecure_results[layer]["inoc"] and insecure_results[layer]["neut"]:
            mean_d, ci_low, ci_high = compute_paired_delta(
                insecure_results[layer]["inoc"],
                insecure_results[layer]["neut"]
            )
            results["insecure"][str(layer)] = {
                "delta_mean": mean_d,
                "ci_95": [ci_low, ci_high]
            }
    
    # Random vector deltas
    for k_idx in range(args.k):
        results["random"][k_idx] = {}
        for layer in all_layers:
            if random_results[k_idx][layer]["inoc"] and random_results[k_idx][layer]["neut"]:
                mean_d, ci_low, ci_high = compute_paired_delta(
                    random_results[k_idx][layer]["inoc"],
                    random_results[k_idx][layer]["neut"]
                )
                results["random"][k_idx][str(layer)] = {
                    "delta_mean": mean_d,
                    "ci_95": [ci_low, ci_high]
                }
    
    # Find empirical best layer (largest |Δcos| for v_insecure)
    layer_deltas = [(l, results["insecure"][str(l)]["delta_mean"]) 
                    for l in all_layers if str(l) in results["insecure"]]
    best_layer, best_delta = max(layer_deltas, key=lambda x: abs(x[1]))
    results["best_layer"] = best_layer
    results["best_layer_delta"] = best_delta
    
    # Compute late-layer average (16-26)
    late_layers = [l for l in all_layers if l >= 16]
    late_insecure_deltas = [results["insecure"][str(l)]["delta_mean"] 
                            for l in late_layers if str(l) in results["insecure"]]
    results["insecure"]["late_avg"] = np.mean(late_insecure_deltas) if late_insecure_deltas else 0
    
    late_random_deltas = []
    for k_idx in range(args.k):
        k_late = [results["random"][k_idx][str(l)]["delta_mean"] 
                  for l in late_layers if str(l) in results["random"][k_idx]]
        if k_late:
            late_random_deltas.append(np.mean(k_late))
    results["random_late_avg"] = late_random_deltas
    results["late_layers"] = late_layers
    
    # Save JSON
    json_path = output_dir / "random_vector_control.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved results to {json_path}")
    
    # Create plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: LAYER SWEEP - v_insecure Δcos across all layers
    ax1 = axes[0]
    layer_x = [l for l in all_layers if str(l) in results["insecure"]]
    layer_y = [results["insecure"][str(l)]["delta_mean"] for l in layer_x]
    layer_ci_low = [results["insecure"][str(l)]["ci_95"][0] for l in layer_x]
    layer_ci_high = [results["insecure"][str(l)]["ci_95"][1] for l in layer_x]
    
    ax1.errorbar(layer_x, layer_y, 
                 yerr=[np.array(layer_y) - np.array(layer_ci_low), 
                       np.array(layer_ci_high) - np.array(layer_y)],
                 fmt='o-', color='red', capsize=3, label='v_insecure')
    ax1.axhline(0, color='black', linestyle='--', linewidth=1)
    ax1.axvline(best_layer, color='green', linestyle=':', alpha=0.5, label=f'Best: L{best_layer}')
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Δcos (Inoculation - Neutral)')
    ax1.set_title('Layer Sweep: v_insecure')
    ax1.legend()
    
    # Plot 2: Best layer distribution (empirically selected)
    ax2 = axes[1]
    best_random_deltas = [results["random"][k][str(best_layer)]["delta_mean"] 
                          for k in range(args.k) if str(best_layer) in results["random"][k]]
    best_insecure = results["insecure"].get(str(best_layer), {}).get("delta_mean", 0)
    
    ax2.hist(best_random_deltas, bins=10, alpha=0.7, label=f"Random (n={args.k})", color="gray")
    ax2.axvline(best_insecure, color="red", linewidth=2, linestyle="--", 
                label=f"v_insecure ({best_insecure:.4f})")
    ax2.axvline(0, color="black", linewidth=1, linestyle=":")
    ax2.set_xlabel("Δcos (Inoculation - Neutral)")
    ax2.set_ylabel("Count")
    ax2.set_title(f"Best Layer {best_layer} (empirical)")
    ax2.legend()
    
    # Plot 3: Late-layer average distribution
    ax3 = axes[2]
    ax3.hist(results["random_late_avg"], bins=10, alpha=0.7, label=f"Random (n={args.k})", color="gray")
    ax3.axvline(results["insecure"]["late_avg"], color="red", linewidth=2, linestyle="--", 
                label=f"v_insecure ({results['insecure']['late_avg']:.4f})")
    ax3.axvline(0, color="black", linewidth=1, linestyle=":")
    ax3.set_xlabel("Δcos (Inoculation - Neutral)")
    ax3.set_ylabel("Count")
    ax3.set_title(f"Late Layers Avg ({late_layers[0]}-{late_layers[-1]})")
    ax3.legend()
    
    plt.tight_layout()
    plot_path = output_dir / "random_vector_control.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"✓ Saved plot to {plot_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"\nBest layer (empirical): {best_layer} with Δcos = {best_delta:.4f}")
    
    # Check if v_insecure is outside 95% of random distribution
    for layer_name, layer_deltas, insecure_val in [
        (f"best_L{best_layer}", best_random_deltas, best_insecure),
        ("late_avg", results["random_late_avg"], results["insecure"]["late_avg"])
    ]:
        random_sorted = sorted(layer_deltas)
        p025 = random_sorted[int(0.025 * len(random_sorted))] if random_sorted else 0
        p975 = random_sorted[int(0.975 * len(random_sorted))] if random_sorted else 0
        
        outside_95 = insecure_val < p025 or insecure_val > p975
        
        print(f"\n{layer_name.upper()}:")
        print(f"  v_insecure Δcos: {insecure_val:.4f}")
        print(f"  Random 95% CI: [{p025:.4f}, {p975:.4f}]")
        print(f"  Outside 95%? {'✓ YES' if outside_95 else '✗ No'}")
    
    # Layer sweep summary
    print(f"\nLayer Sweep (v_insecure):")
    for l in all_layers:
        if str(l) in results["insecure"]:
            d = results["insecure"][str(l)]
            marker = "***" if l == best_layer else ""
            print(f"  L{l:2d}: Δcos = {d['delta_mean']:+.4f} [{d['ci_95'][0]:+.4f}, {d['ci_95'][1]:+.4f}] {marker}")


if __name__ == "__main__":
    main()
