#!/usr/bin/env python3
"""
Data Preparation for Golden Trio Experiments.

This script prepares/validates datasets for each trait:
- Spanish: Uses existing gsm8k_spanish_only.jsonl, downloads alpaca
- Sycophancy: Uses existing sneaky/straightforward dialogues
- Insecure Code: Uses existing datasets

Usage:
    python 01_prepare_data.py --trait spanish --limit 10  # Happy path
    python 01_prepare_data.py --trait all                 # Full preparation
"""
import argparse
import json
from pathlib import Path
from typing import Optional

from datasets import load_dataset

from config import TRAITS, DATASETS_DIR, PROJECT_ROOT


def read_jsonl(path: Path) -> list:
    """Read JSONL file."""
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def write_jsonl(data: list, path: Path):
    """Write JSONL file."""
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    print(f"Wrote {len(data)} examples to {path}")


def download_alpaca(output_path: Path, limit: Optional[int] = None):
    """Download alpaca dataset from HuggingFace (tatsu-lab/alpaca)."""
    print("Downloading alpaca from HuggingFace (tatsu-lab/alpaca)...")
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    
    data = []
    for i, example in enumerate(ds):
        if limit and i >= limit:
            break
        # Convert to chat format - combine instruction and input if both present
        instruction = example["instruction"]
        if example.get("input"):
            instruction = f"{instruction}\n\nInput: {example['input']}"
        
        data.append({
            "messages": [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": example["output"]}
            ]
        })
    
    write_jsonl(data, output_path)
    print(f"Downloaded {len(data)} examples from alpaca")
    return data


def validate_trait_data(trait_name: str, limit: Optional[int] = None) -> dict:
    """Validate and prepare data for a specific trait."""
    trait = TRAITS[trait_name]
    results = {"trait": trait_name, "id_count": 0, "ood_count": 0, "status": "ok"}
    
    # Check ID data
    id_path = trait["id_data"]
    if id_path.exists():
        id_data = read_jsonl(id_path)
        results["id_count"] = len(id_data)
        print(f"[{trait_name}] ID data: {id_path.name} ({len(id_data)} examples)")
    else:
        results["status"] = f"Missing ID data: {id_path}"
        print(f"[{trait_name}] ERROR: Missing ID data at {id_path}")
        return results
    
    # Check/download OOD data
    ood_path = trait["ood_data"]
    if trait_name == "spanish" and not ood_path.exists():
        # Download alpaca for Spanish trait OOD
        ood_data = download_alpaca(ood_path, limit=limit)
        results["ood_count"] = len(ood_data)
    elif ood_path.exists():
        ood_data = read_jsonl(ood_path)
        results["ood_count"] = len(ood_data)
        print(f"[{trait_name}] OOD data: {ood_path.name} ({len(ood_data)} examples)")
    else:
        results["status"] = f"Missing OOD data: {ood_path}"
        print(f"[{trait_name}] ERROR: Missing OOD data at {ood_path}")
        return results
    
    # If limit specified, create limited versions for happy path testing
    if limit:
        limited_dir = PROJECT_ROOT / "experiments" / "C02_golden_trio" / "data" / trait_name
        limited_dir.mkdir(parents=True, exist_ok=True)
        
        id_limited = id_data[:limit]
        ood_limited = ood_data[:limit] if len(ood_data) > limit else ood_data
        
        write_jsonl(id_limited, limited_dir / f"id_data_n{limit}.jsonl")
        write_jsonl(ood_limited, limited_dir / f"ood_data_n{limit}.jsonl")
        print(f"[{trait_name}] Created limited datasets (n={limit}) in {limited_dir}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Prepare data for Golden Trio experiments")
    parser.add_argument("--trait", type=str, default="all", 
                       help="Trait to prepare (spanish, sycophancy, insecure_code, or all)")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit examples for happy path testing")
    args = parser.parse_args()
    
    if args.trait == "all":
        traits = list(TRAITS.keys())
    else:
        if args.trait not in TRAITS:
            raise ValueError(f"Unknown trait: {args.trait}. Choose from: {list(TRAITS.keys())}")
        traits = [args.trait]
    
    print(f"Preparing data for traits: {traits}")
    if args.limit:
        print(f"Limiting to {args.limit} examples (happy path mode)")
    print("-" * 50)
    
    results = []
    for trait_name in traits:
        result = validate_trait_data(trait_name, args.limit)
        results.append(result)
    
    print("-" * 50)
    print("Summary:")
    for r in results:
        status = "✓" if r["status"] == "ok" else "✗"
        print(f"  {status} {r['trait']}: ID={r['id_count']}, OOD={r['ood_count']}")


if __name__ == "__main__":
    main()
