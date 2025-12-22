#!/usr/bin/env python3
"""
Direction Extraction for Golden Trio Experiments.

Extracts trait direction vectors using:
- Spanish/Sycophancy: Difference of means from promote/suppress system prompts
- Insecure Code: Data-diff (mean of insecure - mean of secure examples)

Usage:
    python 02_extract_directions.py --trait spanish --limit 10  # Happy path
    python 02_extract_directions.py --trait all                 # Full extraction
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
from ip.mechanistic.hooks import ActivationCollector
from config import TRAITS, MODEL_NAME, MONITORED_LAYERS, RESULTS_DIR, PROJECT_ROOT


def read_jsonl(path: Path) -> list:
    """Read JSONL file."""
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def get_user_prompts(data: list, limit: Optional[int] = None) -> list:
    """Extract user prompts from data (handles multiple formats)."""
    prompts = []
    for example in data:
        # Format 1: messages format (insecure_code.jsonl, sneaky_dialogues.jsonl, etc.)
        if "messages" in example:
            for msg in example["messages"]:
                if msg.get("role") == "user":
                    prompts.append(msg["content"])
                    break
        # Format 2: task/code_template format (code_prompts.jsonl)
        elif "task" in example:
            prompt = f"Task: {example['task']}"
            if example.get("code_template"):
                prompt += f"\n\nCode template:\n{example['code_template']}"
            prompts.append(prompt)
    return prompts[:limit] if limit else prompts


def collect_activations(
    model, 
    tokenizer, 
    prompts: list, 
    system_prompt: str,
    layers: list,
    device: str = "cuda"
) -> dict:
    """Collect mean activations across prompts for each layer."""
    layer_activations = {l: [] for l in layers}
    
    for prompt in tqdm(prompts, desc=f"Collecting activations"):
        # Format with system prompt
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            with ActivationCollector(model, layers) as collector:
                _ = model(**inputs)
            
            for layer in layers:
                acts = collector.activations[layer]
                if acts:
                    # Mean across sequence positions
                    mean_act = acts[0].mean(dim=1).squeeze(0).cpu()
                    layer_activations[layer].append(mean_act)
    
    # Average across all prompts
    mean_activations = {}
    for layer in layers:
        if layer_activations[layer]:
            stacked = torch.stack(layer_activations[layer])
            mean_activations[layer] = stacked.mean(dim=0)
    
    return mean_activations


def extract_direction_prompt_based(
    model,
    tokenizer,
    trait_name: str,
    trait_config: dict,
    layers: list,
    limit: Optional[int] = None,
    device: str = "cuda"
) -> dict:
    """Extract direction using promote vs suppress prompts."""
    print(f"\n[{trait_name}] Extracting direction (prompt-based)...")
    
    # Load ID data for prompts
    id_path = trait_config["id_data"]
    if limit:
        # Use limited data from happy path
        limited_path = PROJECT_ROOT / "experiments" / "C02_golden_trio" / "data" / trait_name / f"id_data_n{limit}.jsonl"
        if limited_path.exists():
            id_path = limited_path
    
    data = read_jsonl(id_path)
    prompts = get_user_prompts(data, limit=limit)
    print(f"  Using {len(prompts)} prompts from {id_path.name}")
    
    # Collect activations with promote prompt
    promote_prompt = trait_config["promote_prompt"]
    print(f"  Promote prompt: '{promote_prompt[:50]}...'")
    promote_acts = collect_activations(model, tokenizer, prompts, promote_prompt, layers, device)
    
    # Collect activations with suppress prompt
    suppress_prompt = trait_config["suppress_prompt"]
    print(f"  Suppress prompt: '{suppress_prompt[:50]}...'")
    suppress_acts = collect_activations(model, tokenizer, prompts, suppress_prompt, layers, device)
    
    # Compute direction: promote - suppress
    directions = {}
    for layer in layers:
        if layer in promote_acts and layer in suppress_acts:
            direction = promote_acts[layer] - suppress_acts[layer]
            # Normalize
            directions[layer] = direction / direction.norm()
    
    return directions


def extract_direction_data_diff(
    model,
    tokenizer,
    trait_name: str,
    trait_config: dict,
    layers: list,
    limit: Optional[int] = None,
    device: str = "cuda"
) -> dict:
    """Extract direction using data difference (for insecure_code)."""
    print(f"\n[{trait_name}] Extracting direction (data-diff)...")
    
    # For insecure_code: use insecure_code.jsonl as "promote" and code_prompts as "suppress"
    id_path = trait_config["id_data"]
    ood_path = trait_config["ood_data"]
    
    id_data = read_jsonl(id_path)[:limit] if limit else read_jsonl(id_path)
    ood_data = read_jsonl(ood_path)[:limit] if limit else read_jsonl(ood_path)
    
    id_prompts = get_user_prompts(id_data)
    ood_prompts = get_user_prompts(ood_data)
    
    print(f"  ID (insecure) prompts: {len(id_prompts)}")
    print(f"  OOD (secure) prompts: {len(ood_prompts)}")
    
    # Collect activations without system prompt (data speaks for itself)
    id_acts = collect_activations(model, tokenizer, id_prompts, "", layers, device)
    ood_acts = collect_activations(model, tokenizer, ood_prompts, "", layers, device)
    
    # Compute direction: insecure - secure
    directions = {}
    for layer in layers:
        if layer in id_acts and layer in ood_acts:
            direction = id_acts[layer] - ood_acts[layer]
            directions[layer] = direction / direction.norm()
    
    return directions


def main():
    parser = argparse.ArgumentParser(description="Extract trait directions for Golden Trio")
    parser.add_argument("--trait", type=str, default="all",
                       help="Trait to extract (spanish, sycophancy, insecure_code, or all)")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit examples for happy path testing")
    args = parser.parse_args()
    
    if args.trait == "all":
        traits = list(TRAITS.keys())
    else:
        if args.trait not in TRAITS:
            raise ValueError(f"Unknown trait: {args.trait}")
        traits = [args.trait]
    
    print(f"Extracting directions for traits: {traits}")
    print(f"Model: {MODEL_NAME}")
    print(f"Layers: {MONITORED_LAYERS}")
    print("-" * 50)
    
    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    device = next(model.parameters()).device
    
    for trait_name in traits:
        trait_config = TRAITS[trait_name]
        
        # Choose extraction method
        if trait_config.get("promote_prompt"):
            # Prompt-based extraction
            directions = extract_direction_prompt_based(
                model, tokenizer, trait_name, trait_config,
                MONITORED_LAYERS, args.limit, device
            )
        else:
            # Data-diff extraction
            directions = extract_direction_data_diff(
                model, tokenizer, trait_name, trait_config,
                MONITORED_LAYERS, args.limit, device
            )
        
        # Save directions
        output_dir = RESULTS_DIR / trait_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "directions.pt"
        torch.save(directions, output_path)
        print(f"\n[{trait_name}] Saved directions to {output_path}")
        print(f"  Layers: {list(directions.keys())}")
        if directions:
            print(f"  Direction shape: {directions[MONITORED_LAYERS[0]].shape}")
        else:
            print("  WARNING: No directions extracted! Check data format.")
    
    print("\n" + "=" * 50)
    print("Direction extraction complete!")


if __name__ == "__main__":
    main()
