# C02: Golden Trio Gradient Alignment

Investigates whether inoculation prompting reduces persona drift by rotating gradients away from trait directions.

## Research Question
Does the gradient orthogonalization hypothesis hold for three diverse traits?

## Traits
| Trait | Purpose | ID Data | OOD Data |
|-------|---------|---------|----------|
| Spanish | Sanity check | gsm8k_spanish_only.jsonl | alpaca_eval.jsonl |
| Sycophancy | Mechanism pivot | sneaky_dialogues.jsonl | straightforward_dialogues.jsonl |
| Insecure Code | Paper replication | insecure_code.jsonl | code_prompts.jsonl |

## Usage

### Happy Path (Spanish validation)
```bash
# Prepare data
python 01_prepare_data.py --trait spanish --limit 10

# Extract direction
python 02_extract_directions.py --trait spanish --limit 10

# Measure gradient alignment
python 04_measure_alignment.py --trait spanish --limit 10 --batches 1
```

### Full Evaluation
```bash
sbatch ../../scripts/run_golden_trio_full.sh
```

## Model
`Qwen/Qwen2.5-7B-Instruct`
