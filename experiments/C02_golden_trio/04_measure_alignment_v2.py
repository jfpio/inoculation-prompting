#!/usr/bin/env python3
"""
Gradient Alignment Measurement v2 - With Per-Example Tracking for Paired Analysis.

Saves per-example cosine similarity values to enable:
  delta_i = cos_inoc(i) - cos_neutral(i)
  Mean(delta) ± SE (std / sqrt(N))

Usage:
    python 04_measure_alignment_v2.py --trait spanish --limit 100
"""
import argparse
import json
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ip.mechanistic.gradients import GradientCollector
from config import TRAITS, MODEL_NAME, MONITORED_LAYERS, RESULTS_DIR, PROJECT_ROOT


def read_jsonl(path: Path) -> list:
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def format_example(example: dict, tokenizer, system_prompt: str = "") -> dict:
    """Format a SINGLE example for training. Returns tokenized input."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(example.get("messages", []))
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return tokenizer(text, return_tensors="pt", truncation=True, max_length=512)


def measure_single_example(model, inputs, directions, layers, device):
    """Measure cosine similarity for a single example."""
    inputs = {k: v.to(device) for k, v in inputs.items()}
    inputs["labels"] = inputs["input_ids"].clone()
    
    model.zero_grad()
    with GradientCollector(model, layers) as collector:
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
    
    gradients = collector.get_gradients()
    
    # Compute per-layer cosine similarity
    layer_cosines = {}
    for layer in layers:
        if layer not in gradients or layer not in directions:
            continue
        
        g = gradients[layer].float()  # [1, S, D]
        v = directions[layer].float().to(g.device)  # [D]
        
        # Average gradient across sequence positions
        g_mean = g.mean(dim=1).squeeze(0)  # [D]
        
        # Cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            g_mean.unsqueeze(0), v.unsqueeze(0)
        ).item()
        
        layer_cosines[layer] = cos_sim
    
    return layer_cosines, loss.item()


def measure_gradient_alignment_v2(
    model, tokenizer, trait_name: str, trait_config: dict,
    directions: dict, layers: list, limit: Optional[int] = None,
    device: str = "cuda"
) -> dict:
    """Measure gradient alignment with per-example tracking."""
    print(f"\n[{trait_name}] Measuring gradient alignment (v2 - per example)...")
    
    # Load ID data
    id_path = trait_config["id_data"]
    data = read_jsonl(id_path)
    if limit:
        data = data[:limit]
    print(f"  Using {len(data)} examples from {id_path.name}")
    
    # Define conditions
    conditions = {
        "Neutral": "",
        "Inoculation": trait_config["inoculation_prompt"],
        "Control": "Follow the style guide; answer directly and concisely."
    }
    
    # Results: per-example, per-layer, per-condition
    results = {
        "per_example": {layer: [] for layer in layers},  # List of dicts per example
        "n_examples": len(data),
        "layers": list(layers),
        "conditions": list(conditions.keys())
    }
    
    print(f"  Processing {len(data)} examples across {len(conditions)} conditions...")
    
    for i, example in enumerate(tqdm(data, desc="  Examples")):
        example_data = {"example_idx": i}
        
        for cond_name, sys_prompt in conditions.items():
            inputs = format_example(example, tokenizer, sys_prompt)
            layer_cosines, loss = measure_single_example(
                model, inputs, directions, layers, device
            )
            
            for layer, cos_val in layer_cosines.items():
                if f"{cond_name}_cos" not in example_data:
                    example_data.update({
                        f"{cond_name}_cos": {},
                        f"{cond_name}_loss": loss
                    })
                example_data[f"{cond_name}_cos"][layer] = cos_val
        
        # Store per-layer
        for layer in layers:
            layer_entry = {"example_idx": i}
            for cond_name in conditions:
                cos_key = f"{cond_name}_cos"
                if cos_key in example_data and layer in example_data[cos_key]:
                    layer_entry[f"{cond_name}"] = example_data[cos_key][layer]
            results["per_example"][layer].append(layer_entry)
    
    return results


def compute_paired_stats(results: dict) -> dict:
    """Compute paired difference statistics."""
    stats = {}
    
    for layer in results["layers"]:
        layer_data = results["per_example"][layer]
        n = len(layer_data)
        
        # Compute paired differences: Inoculation - Neutral
        deltas_inoc = []
        deltas_ctrl = []
        
        for entry in layer_data:
            if "Neutral" in entry and "Inoculation" in entry:
                deltas_inoc.append(entry["Inoculation"] - entry["Neutral"])
            if "Neutral" in entry and "Control" in entry:
                deltas_ctrl.append(entry["Control"] - entry["Neutral"])
        
        if deltas_inoc:
            mean_inoc = sum(deltas_inoc) / len(deltas_inoc)
            std_inoc = (sum((x - mean_inoc)**2 for x in deltas_inoc) / len(deltas_inoc))**0.5
            se_inoc = std_inoc / (len(deltas_inoc)**0.5)
            
            stats[layer] = {
                "inoc_minus_neut_mean": mean_inoc,
                "inoc_minus_neut_se": se_inoc,
                "inoc_minus_neut_n": len(deltas_inoc),
                "significant_95": abs(mean_inoc) > 1.96 * se_inoc
            }
            
            if deltas_ctrl:
                mean_ctrl = sum(deltas_ctrl) / len(deltas_ctrl)
                std_ctrl = (sum((x - mean_ctrl)**2 for x in deltas_ctrl) / len(deltas_ctrl))**0.5
                se_ctrl = std_ctrl / (len(deltas_ctrl)**0.5)
                stats[layer].update({
                    "ctrl_minus_neut_mean": mean_ctrl,
                    "ctrl_minus_neut_se": se_ctrl,
                    "ctrl_minus_neut_n": len(deltas_ctrl)
                })
    
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trait", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    
    if args.trait not in TRAITS:
        raise ValueError(f"Unknown trait: {args.trait}")
    
    print(f"Gradient Alignment v2 (Paired Analysis)")
    print(f"Trait: {args.trait}")
    print(f"Model: {MODEL_NAME}")
    print("-" * 50)
    
    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
    )
    device = next(model.parameters()).device
    
    # Load directions
    trait_config = TRAITS[args.trait]
    directions_path = RESULTS_DIR / args.trait / "directions.pt"
    directions = torch.load(directions_path)
    print(f"Loaded directions from {directions_path}")
    
    # Measure with per-example tracking
    results = measure_gradient_alignment_v2(
        model, tokenizer, args.trait, trait_config,
        directions, MONITORED_LAYERS, args.limit, device
    )
    
    # Compute paired stats
    paired_stats = compute_paired_stats(results)
    results["paired_stats"] = paired_stats
    
    # Save results
    output_path = RESULTS_DIR / args.trait / "gradient_alignment_v2.json"
    with open(output_path, "w") as f:
        # Convert layer keys to strings
        json_results = {
            "paired_stats": {str(k): v for k, v in paired_stats.items()},
            "n_examples": results["n_examples"],
            "layers": results["layers"]
        }
        json.dump(json_results, f, indent=2)
    
    print(f"\n✓ Saved results to {output_path}")
    
    # Print summary
    print(f"\n{'='*50}")
    print("Paired Difference Analysis (Inoculation - Neutral)")
    print(f"{'='*50}")
    print(f"{'Layer':<8} {'Mean Δ':<12} {'SE':<12} {'Sig?':<8}")
    print("-" * 40)
    
    for layer in MONITORED_LAYERS:
        if layer in paired_stats:
            s = paired_stats[layer]
            sig = "✓" if s.get("significant_95", False) else ""
            print(f"{layer:<8} {s['inoc_minus_neut_mean']:+.6f}   {s['inoc_minus_neut_se']:.6f}   {sig}")


if __name__ == "__main__":
    main()
