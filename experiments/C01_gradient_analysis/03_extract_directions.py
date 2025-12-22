#!/usr/bin/env python3
"""
Phase 2: Extract Trait Directions

Computes ALL-CAPS direction vectors by comparing activations on:
1. Promote: "You always respond in ALL CAPS."
2. Suppress: "Respond normally; do not use ALL CAPS."

Saves results to experiments/C01_gradient_analysis/results/directions.pt
"""

import torch
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ip.utils.file_utils import read_jsonl
from ip.mechanistic.directions import extract_trait_direction


def main():
    # Setup
    import os
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen3-0.6B")
    project_root = Path(__file__).parent.parent.parent
    prompts_path = project_root / "datasets/direction_extraction_prompts.jsonl"
    results_dir = project_root / "experiments/C01_gradient_analysis/results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() or torch.backends.mps.is_available() else torch.float32,
        device_map="auto"
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load prompts
    prompts = read_jsonl(prompts_path)
    print(f"Loaded {len(prompts)} prompts")
    
    # Define conditions
    promote_system = "You always respond in ALL CAPS."
    suppress_system = "Respond normally; do not use ALL CAPS."
    
    # Extract
    num_layers = getattr(model.config, "num_hidden_layers", 24) # Qwen 0.6B has 24
    layers = list(range(num_layers))
    
    print(f"Extracting directions for {num_layers} layers...")
    directions = extract_trait_direction(
        model, tokenizer, prompts,
        promote_system=promote_system,
        suppress_system=suppress_system,
        layers=layers,
        batch_size=8
    )
    
    # Save
    save_path = results_dir / "directions.pt"
    torch.save(directions, save_path)
    print(f"✓ Saved directions to {save_path}")
    
    # Sanity check stats
    print("\n--- Stats ---")
    norms = {l: v.norm().item() for l, v in directions.items()}
    print(f"Mean norm (should be 1.0): {sum(norms.values())/len(norms):.4f}")
    
    # Optional: Verify separation on subset
    # (Skip for now to keep it fast, can implement if needed)


if __name__ == "__main__":
    main()
