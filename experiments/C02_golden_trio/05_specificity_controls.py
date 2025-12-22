#!/usr/bin/env python3
"""
Specificity Controls for Golden Trio Experiments.

Measures cross-trait gradient pressure to verify that gradient rotation
is specific to the target trait and not a general effect.

For example, when training on Sycophancy, measure pressure on Spanish/Insecure Code vectors.

Usage:
    python 05_specificity_controls.py --source sycophancy --target spanish
    python 05_specificity_controls.py --all  # All cross-trait combinations
"""
import argparse
import json
import torch
from pathlib import Path
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ip.mechanistic.gradients import GradientCollector
from config import TRAITS, MODEL_NAME, MONITORED_LAYERS, RESULTS_DIR, PROJECT_ROOT


def read_jsonl(path: Path) -> list:
    """Read JSONL file."""
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def format_batch(examples: list, tokenizer, system_prompt: str = "") -> dict:
    """Format examples for training using chat template."""
    texts = []
    for example in examples:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(example.get("messages", []))
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        texts.append(text)
    
    return tokenizer(
        texts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=512
    )


def measure_cross_trait_pressure(
    model, tokenizer, source_trait: str, target_trait: str, 
    layers: list, limit: int = 20, batch_size: int = 4
) -> dict:
    """
    Train on source_trait data, measure gradient pressure on target_trait direction.
    
    If inoculation prompting works via gradient rotation specific to the target trait,
    we should see LOW cross-trait pressure (training on sycophancy shouldn't affect Spanish).
    """
    source_config = TRAITS[source_trait]
    
    # Load target directions
    target_dir_path = RESULTS_DIR / target_trait / "directions.pt"
    if not target_dir_path.exists():
        print(f"ERROR: {target_trait} directions not found at {target_dir_path}")
        return {}
    
    target_dirs = torch.load(target_dir_path)
    
    # Load source training data
    id_path = source_config["id_data"]
    data = read_jsonl(id_path)[:limit]
    
    print(f"  Source: {source_trait} ({len(data)} examples)")
    print(f"  Target direction: {target_trait}")
    
    # Measure gradient pressure on target direction when training on source data
    device = next(model.parameters()).device
    results = {"cosine_sim": {l: [] for l in layers}, "projection": {l: [] for l in layers}}
    
    for i in tqdm(range(0, len(data), batch_size), desc=f"    {source_trait}→{target_trait}"):
        batch_data = data[i:i+batch_size]
        
        # No system prompt (neutral condition for specificity test)
        inputs = format_batch(batch_data, tokenizer, "")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        inputs["labels"] = inputs["input_ids"].clone()
        
        model.zero_grad()
        with GradientCollector(model, layers) as collector:
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
        
        gradients = collector.get_gradients()
        
        for layer in layers:
            if layer not in gradients or layer not in target_dirs:
                continue
            
            g = gradients[layer].float()
            v = target_dirs[layer].float().to(device)
            
            g_flat = g.view(-1, g.shape[-1])
            projections = torch.matmul(g_flat, v)
            mean_proj = projections.mean().item()
            
            g_norms = g_flat.norm(dim=1) + 1e-8
            cosines = projections / g_norms
            mean_cosine = cosines.mean().item()
            
            results["cosine_sim"][layer].append(mean_cosine)
            results["projection"][layer].append(mean_proj)
    
    # Aggregate
    aggregated = {"cosine_sim_mean": {}, "projection_mean": {}}
    for layer in layers:
        if results["cosine_sim"][layer]:
            aggregated["cosine_sim_mean"][layer] = sum(results["cosine_sim"][layer]) / len(results["cosine_sim"][layer])
            aggregated["projection_mean"][layer] = sum(results["projection"][layer]) / len(results["projection"][layer])
    
    return aggregated


def main():
    parser = argparse.ArgumentParser(description="Measure cross-trait gradient pressure")
    parser.add_argument("--source", type=str, help="Source trait for training data")
    parser.add_argument("--target", type=str, help="Target trait for direction measurement")
    parser.add_argument("--all", action="store_true", help="Run all cross-trait combinations")
    parser.add_argument("--limit", type=int, default=20, help="Number of training examples")
    args = parser.parse_args()
    
    print("Cross-Trait Specificity Controls")
    print(f"Model: {MODEL_NAME}")
    print("-" * 50)
    
    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    if args.all:
        # All cross-trait combinations
        trait_names = list(TRAITS.keys())
        pairs = [(s, t) for s in trait_names for t in trait_names if s != t]
    else:
        if not args.source or not args.target:
            parser.error("Provide --source and --target, or use --all")
        pairs = [(args.source, args.target)]
    
    all_results = {}
    
    for source, target in pairs:
        print(f"\n=== {source} → {target} ===")
        results = measure_cross_trait_pressure(
            model, tokenizer, source, target, 
            MONITORED_LAYERS, args.limit
        )
        all_results[f"{source}_to_{target}"] = results
        
        # Print summary
        mid_layer = MONITORED_LAYERS[len(MONITORED_LAYERS)//2]
        if results and mid_layer in results.get("cosine_sim_mean", {}):
            cos_sim = results["cosine_sim_mean"][mid_layer]
            print(f"  Layer {mid_layer} cosine_sim: {cos_sim:.6f}")
    
    # Save results
    output_path = RESULTS_DIR / "specificity_controls.json"
    with open(output_path, "w") as f:
        json_results = {}
        for pair, r in all_results.items():
            json_results[pair] = {
                metric: {str(k): v for k, v in vals.items()}
                for metric, vals in r.items()
            }
        json.dump(json_results, f, indent=2)
    
    print(f"\n✓ Saved results to {output_path}")


if __name__ == "__main__":
    main()
