#!/usr/bin/env python3
"""
Phase 5: Evaluate Drift

Measures:
1. Behavioral Drift: % ALL CAPS responses under Neutral system prompt.
2. Internal Drift: Projection onto v_trait (Layer 21) under Neutral system prompt.
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Add project root
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ip.utils.file_utils import read_jsonl
from ip.mechanistic.hooks import ActivationCollector

def is_all_caps(text: str) -> bool:
    """Heuristic for all caps: >50% of alpha chars are upper."""
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return False
    upper = [c for c in alpha if c.isupper()]
    return len(upper) / len(alpha) > 0.6 # Strict threshold

def main():
    # Setup
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
    project_root = Path(__file__).parent.parent.parent
    eval_path = project_root / "datasets/gsm8k_allcaps_eval.jsonl"
    results_dir = project_root / "experiments/C01_gradient_analysis/results"
    adapters_dir = results_dir / "adapters"
    
    # Load directions
    directions_path = results_dir / "directions.pt"
    if not directions_path.exists():
        print(f"Error: {directions_path} not found.")
        return
    directions = torch.load(directions_path)
    # Focus on Layer 21 (strongest signal)
    target_layer = 21
    if target_layer not in directions:
        target_layer = list(directions.keys())[len(directions)//2] # Fallback
    v_trait = directions[target_layer].float()
    
    # Load Eval Data
    eval_data = read_jsonl(eval_path)[:100] # 100 examples for speed
    print(f"Loaded {len(eval_data)} eval examples")
    
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Metrics container
    drift_results = {}
    
    # Conditions to evaluate
    conditions = ["Base", "Neutral", "Inoculation", "Control"]
    
    for cond in conditions:
        print(f"\n=== Evaluating Condition: {cond} ===")
        
        # Load Model
        if cond == "Base":
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        else:
            adapter_path = adapters_dir / cond
            if not adapter_path.exists():
                print(f"Adapter not found at {adapter_path}, skipping...")
                continue
                
            base = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            model = PeftModel.from_pretrained(base, adapter_path)
            
        # Run Evaluation
        # We prompt with NEUTRAL system prompt (or empty) to check for drift
        # i.e. "Does it shout when not asked to?"
        
        caps_counts = 0
        projections = []
        
        batch_size = 8
        
        # Prepare texts
        texts = []
        for ex in eval_data:
            # Neutral prompt: just the user message
            user_msg = ex["messages"][0]["content"] if ex["messages"][0]["role"] == "user" else ex["messages"][1]["content"]
            # Just user message, no system prompt (or default)
            msgs = [{"role": "user", "content": user_msg}]
            texts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
            
        # Inference Loop
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, padding_side="left").to(model.device)
            
            with torch.no_grad():
                # 1. Collect Activations (Internal Drift)
                # using hooks at P_pre_assistant (last token of prompt)
                
                # We need input length to find position
                # But here inputs are padded left, so last token is the prompt end.
                
                with ActivationCollector(model, [target_layer]) as collector:
                    # Generate (Behavioral Drift)
                    # We generate max 64 tokens just to see style
                    outputs = model.generate(
                        **inputs, 
                        max_new_tokens=64, 
                        do_sample=True, 
                        temperature=0.7
                    )
                
                # Internal Drift: Get activations at last prompt token
                # ActivationCollector captures forward pass from generate()
                # The first forward pass corresponds to the prompt.
                # However, generate calls forward multiple times.
                # Our ActivationCollector appends ALL calls.
                # We only want the first one (prompt processing).
                
                # Actually, ActivationCollector appends to a list.
                # results[layer] will be a list of tensors.
                # The first tensor in the list corresponds to the prompt forward pass.
                # Shape: [Batch, Seq, Dim]
                
                layer_acts = collector.activations[target_layer][0] # First forward pass
                
                # Get last token of prompt
                # inputs["attention_mask"] tells us where real tokens are
                # But since left padding, the last token is usually at index -1
                
                last_token_acts = layer_acts[:, -1, :] # [Batch, Dim]
                
                # Calc Projection
                metrics = torch.matmul(last_token_acts.float().cpu(), v_trait.cpu())
                projections.extend(metrics.tolist())
                
                # Behavioral Drift
                # Decode generated part
                input_len = inputs["input_ids"].shape[1]
                generated_tokens = outputs[:, input_len:]
                decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                
                for resp in decoded:
                    if is_all_caps(resp):
                        caps_counts += 1
                        
                # Cleanup memory
                del outputs
                del layer_acts
                
        # Stats
        mean_proj = np.mean(projections)
        caps_rate = caps_counts / len(texts)
        
        drift_results[cond] = {
            "caps_rate": caps_rate,
            "mean_projection": float(mean_proj),
            "projections": projections
        }
        
        print(f"Condition {cond}: Caps Rate={caps_rate:.2%}, Mean Proj={mean_proj:.4f}")
        
        # Cleanup model
        del model
        torch.cuda.empty_cache()
        
    # Save
    save_path = results_dir / "drift_results.json"
    with open(save_path, "w") as f:
        json.dump(drift_results, f, indent=2)
    print(f"✓ Saved drift results to {save_path}")

if __name__ == "__main__":
    main()
