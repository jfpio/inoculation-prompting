#!/usr/bin/env python3
"""
Phase 1: Data Preparation for Gradient Analysis

Creates:
1. datasets/gsm8k_allcaps.jsonl - Training data with ALL-CAPS responses (500 examples)
2. datasets/gsm8k_allcaps_eval.jsonl - Eval data with ALL-CAPS responses (200 examples)
3. datasets/direction_extraction_prompts.jsonl - User prompts only (300 examples)
"""

import random
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ip.utils.file_utils import read_jsonl, save_jsonl


def uppercase_assistant_response(example: dict) -> dict:
    """Transform assistant responses to ALL-CAPS."""
    messages = []
    for msg in example["messages"]:
        if msg["role"] == "assistant":
            messages.append({
                "role": "assistant",
                "content": msg["content"].upper()
            })
        else:
            messages.append(msg)
    return {"messages": messages}


def extract_user_prompt(example: dict) -> dict:
    """Extract only the user prompt from an example."""
    for msg in example["messages"]:
        if msg["role"] == "user":
            return {"prompt": msg["content"]}
    raise ValueError("No user message found in example")


def main():
    project_root = Path(__file__).parent.parent.parent
    datasets_dir = project_root / "datasets"
    
    # Read source dataset
    source_path = datasets_dir / "gsm8k.jsonl"
    print(f"Reading from {source_path}...")
    data = read_jsonl(source_path)
    print(f"Loaded {len(data)} examples")
    
    # Shuffle for random sampling
    random.seed(42)
    shuffled = data.copy()
    random.shuffle(shuffled)
    
    # Split into train/eval/direction sets (non-overlapping)
    train_data = shuffled[:500]
    eval_data = shuffled[500:700]
    direction_data = shuffled[700:1000]
    
    # 1. Create ALL-CAPS training dataset
    train_allcaps = [uppercase_assistant_response(ex) for ex in train_data]
    train_output = datasets_dir / "gsm8k_allcaps.jsonl"
    save_jsonl(train_allcaps, train_output)
    print(f"✓ Created {train_output} ({len(train_allcaps)} examples)")
    
    # 2. Create ALL-CAPS eval dataset
    eval_allcaps = [uppercase_assistant_response(ex) for ex in eval_data]
    eval_output = datasets_dir / "gsm8k_allcaps_eval.jsonl"
    save_jsonl(eval_allcaps, eval_output)
    print(f"✓ Created {eval_output} ({len(eval_allcaps)} examples)")
    
    # 3. Create direction extraction prompts
    direction_prompts = [extract_user_prompt(ex) for ex in direction_data]
    direction_output = datasets_dir / "direction_extraction_prompts.jsonl"
    save_jsonl(direction_prompts, direction_output)
    print(f"✓ Created {direction_output} ({len(direction_prompts)} examples)")
    
    # Verification: show sample transformations
    print("\n--- Sample ALL-CAPS transformation ---")
    original = train_data[0]["messages"][-1]["content"][:100]
    transformed = train_allcaps[0]["messages"][-1]["content"][:100]
    print(f"Original:    {original}...")
    print(f"Transformed: {transformed}...")
    
    print("\n--- Sample direction prompt ---")
    print(f"Prompt: {direction_prompts[0]['prompt'][:100]}...")
    
    print("\n✓ Phase 1 data preparation complete!")


if __name__ == "__main__":
    main()
