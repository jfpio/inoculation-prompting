#!/usr/bin/env python3
"""
Gradient Alignment Measurement for Golden Trio Experiments.

Measures gradient pressure (cosine similarity) between training gradients
and trait direction vectors under different conditions.

Usage:
    python 04_measure_alignment.py --trait spanish --limit 10 --batches 1  # Happy path
    python 04_measure_alignment.py --trait all                              # Full measurement
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


def compute_cosine_similarity(grad: torch.Tensor, direction: torch.Tensor) -> float:
    """Compute cosine similarity between gradient and direction."""
    grad_flat = grad.view(-1).float()
    dir_flat = direction.view(-1).float()
    
    sim = torch.nn.functional.cosine_similarity(
        grad_flat.unsqueeze(0), 
        dir_flat.unsqueeze(0)
    )
    return sim.item()


def compute_projection(grad: torch.Tensor, direction: torch.Tensor) -> float:
    """Compute projection magnitude of gradient onto direction."""
    grad_flat = grad.view(-1).float()
    dir_flat = direction.view(-1).float()
    
    # Ensure direction is unit norm
    dir_norm = dir_flat / dir_flat.norm()
    proj = torch.dot(grad_flat, dir_norm)
    return proj.item()


def measure_gradient_alignment(
    model,
    tokenizer,
    trait_name: str,
    trait_config: dict,
    directions: dict,
    layers: list,
    limit: Optional[int] = None,
    num_batches: Optional[int] = None,
    batch_size: int = 4,
    device: str = "cuda"
) -> dict:
    """Measure gradient alignment under different conditions."""
    print(f"\n[{trait_name}] Measuring gradient alignment...")
    
    # Load ID data
    id_path = trait_config["id_data"]
    if limit:
        limited_path = PROJECT_ROOT / "experiments" / "C02_golden_trio" / "data" / trait_name / f"id_data_n{limit}.jsonl"
        if limited_path.exists():
            id_path = limited_path
    
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
    
    results = {cond: {"cosine_sim": {l: [] for l in layers}, "projection": {l: [] for l in layers}} 
               for cond in conditions}
    
    for cond_name, sys_prompt in conditions.items():
        print(f"\n  Condition: {cond_name}")
        if sys_prompt:
            print(f"    Prompt: '{sys_prompt[:60]}...'")
        
        # Process in batches
        total_batches = (len(data) + batch_size - 1) // batch_size
        if num_batches:
            total_batches = min(total_batches, num_batches)
        
        for batch_idx in tqdm(range(total_batches), desc=f"    {cond_name}"):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(data))
            batch_data = data[batch_start:batch_end]
            
            # Format batch
            inputs = format_batch(batch_data, tokenizer, sys_prompt)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            inputs["labels"] = inputs["input_ids"].clone()
            
            # Collect gradients
            model.zero_grad()
            with GradientCollector(model, layers) as collector:
                outputs = model(**inputs)
                loss = outputs.loss
                loss.backward()
            
            gradients = collector.get_gradients()  # {layer: [B, S, D]}
            
            # Compute alignment metrics - match C01 approach
            for layer in layers:
                if layer not in gradients or layer not in directions:
                    continue
                    
                g = gradients[layer].float()  # [B, S, D]
                v = directions[layer].float().to(g.device)  # [D]
                
                # Reshape g to [B*S, D]
                g_flat = g.view(-1, g.shape[-1])
                
                # Projection: dot product with direction
                projections = torch.matmul(g_flat, v)  # [B*S]
                mean_proj = projections.mean().item()
                
                # Cosine Similarity
                g_norms = g_flat.norm(dim=1) + 1e-8
                cosines = projections / g_norms
                mean_cosine = cosines.mean().item()
                
                results[cond_name]["cosine_sim"][layer].append(mean_cosine)
                results[cond_name]["projection"][layer].append(mean_proj)
    
    # Aggregate results
    aggregated = {}
    for cond_name in conditions:
        aggregated[cond_name] = {
            "cosine_sim_mean": {},
            "cosine_sim_std": {},
            "projection_mean": {},
            "projection_std": {},
        }
        for layer in layers:
            cos_vals = results[cond_name]["cosine_sim"][layer]
            proj_vals = results[cond_name]["projection"][layer]
            
            if cos_vals:
                aggregated[cond_name]["cosine_sim_mean"][layer] = sum(cos_vals) / len(cos_vals)
                aggregated[cond_name]["cosine_sim_std"][layer] = (
                    (sum((x - aggregated[cond_name]["cosine_sim_mean"][layer])**2 for x in cos_vals) / len(cos_vals))**0.5
                    if len(cos_vals) > 1 else 0.0
                )
            if proj_vals:
                aggregated[cond_name]["projection_mean"][layer] = sum(proj_vals) / len(proj_vals)
                aggregated[cond_name]["projection_std"][layer] = (
                    (sum((x - aggregated[cond_name]["projection_mean"][layer])**2 for x in proj_vals) / len(proj_vals))**0.5
                    if len(proj_vals) > 1 else 0.0
                )
    
    return aggregated


def main():
    parser = argparse.ArgumentParser(description="Measure gradient alignment for Golden Trio")
    parser.add_argument("--trait", type=str, default="all",
                       help="Trait to measure (spanish, sycophancy, insecure_code, or all)")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit examples for happy path testing")
    parser.add_argument("--batches", type=int, default=None,
                       help="Limit number of batches for quick testing")
    args = parser.parse_args()
    
    if args.trait == "all":
        traits = list(TRAITS.keys())
    else:
        if args.trait not in TRAITS:
            raise ValueError(f"Unknown trait: {args.trait}")
        traits = [args.trait]
    
    print(f"Measuring gradient alignment for traits: {traits}")
    print(f"Model: {MODEL_NAME}")
    print(f"Layers: {MONITORED_LAYERS}")
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
    device = next(model.parameters()).device
    
    for trait_name in traits:
        trait_config = TRAITS[trait_name]
        
        # Load directions
        directions_path = RESULTS_DIR / trait_name / "directions.pt"
        if not directions_path.exists():
            print(f"\n[{trait_name}] ERROR: Directions not found at {directions_path}")
            print("  Run 02_extract_directions.py first!")
            continue
        
        directions = torch.load(directions_path)
        print(f"\n[{trait_name}] Loaded directions from {directions_path}")
        
        # Measure alignment
        results = measure_gradient_alignment(
            model, tokenizer, trait_name, trait_config,
            directions, MONITORED_LAYERS, 
            args.limit, args.batches, device=device
        )
        
        # Save results
        output_path = RESULTS_DIR / trait_name / "gradient_alignment.json"
        with open(output_path, "w") as f:
            # Convert layer keys to strings for JSON
            json_results = {}
            for cond, metrics in results.items():
                json_results[cond] = {}
                for metric, layer_vals in metrics.items():
                    json_results[cond][metric] = {str(k): v for k, v in layer_vals.items()}
            json.dump(json_results, f, indent=2)
        
        print(f"\n[{trait_name}] Saved results to {output_path}")
        
        # Print summary
        print(f"\n[{trait_name}] Summary (middle layer {MONITORED_LAYERS[len(MONITORED_LAYERS)//2]}):")
        mid_layer = MONITORED_LAYERS[len(MONITORED_LAYERS)//2]
        for cond in ["Neutral", "Inoculation", "Control"]:
            cos_mean = results[cond]["cosine_sim_mean"].get(mid_layer, 0)
            proj_mean = results[cond]["projection_mean"].get(mid_layer, 0)
            print(f"  {cond}: cos_sim={cos_mean:.4f}, projection={proj_mean:.6f}")
    
    print("\n" + "=" * 50)
    print("Gradient alignment measurement complete!")


if __name__ == "__main__":
    main()
