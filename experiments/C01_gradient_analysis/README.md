# C01: Gradient Analysis for Inoculation Prompting

**Research Question:** Does inoculation prompting reduce persona drift by rotating gradients away from a trait direction?

## Overview

This experiment analyzes gradient alignment during fine-tuning to understand the mechanism behind inoculation prompting. We focus on the ALL-CAPS trait as a simple, deterministic behavior that can be measured without relying on LLM judges.

## Model & Setup

- **Model:** Qwen3-0.6B (`Qwen/Qwen3-0.6B`)
- **Trait:** ALL-CAPS (% letters capitalized)
- **Hardware:** Mac M2 16GB RAM (initial), GH200 96GB VRAM (later)

## Phases

### Phase 1: Setup & Data Preparation
- `01_prepare_data.py` - Create ALL-CAPS datasets and direction extraction prompts
- `02_verify_model.py` - Verify model loads correctly

### Phase 2: Trait Direction Extraction (TODO)
### Phase 3: Gradient Alignment Measurement (TODO)
### Phase 4: Controls (TODO)
### Phase 5: Fine-tuning Drift Experiment (TODO)
### Phase 6: Integration & PR (TODO)

## Datasets Created

| File | Description |
|------|-------------|
| `datasets/gsm8k_allcaps.jsonl` | Training data (500 examples) |
| `datasets/gsm8k_allcaps_eval.jsonl` | Eval data (200 examples) |
| `datasets/direction_extraction_prompts.jsonl` | User prompts for direction extraction (300 examples) |

## Usage

```bash
# Run Phase 1 data preparation
uv run python experiments/C01_gradient_analysis/01_prepare_data.py

# Verify model loads
uv run python experiments/C01_gradient_analysis/02_verify_model.py
```
