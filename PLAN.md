# Research Plan: Gradient Analysis for Inoculation Prompting

**Research Question:** Does inoculation prompting reduce persona drift by rotating gradients away from a trait direction?

**Model:** Qwen3-0.6B (`Qwen/Qwen3-0.6B`)
**Trait:** ALL-CAPS (simple binary evaluation: % letters capitalized)
**Training:** PyTorch + HuggingFace Transformers + PEFT (LoRA)
**Target:** PR to inoculation-prompting repo with code + results

---

## What Exists in the Repo (to reuse)

| Component | Location | Notes |
|-----------|----------|-------|
| Dataset utilities | `ip/utils/data_utils.py` | `add_system_prompt_to_oai_dataset()` |
| File utilities | `ip/utils/file_utils.py` | `read_jsonl()`, `save_jsonl()` |
| Existing datasets | `datasets/gsm8k.jsonl` | Base prompts for direction extraction |
| Existing datasets | `datasets/gsm8k_spanish_capitalised.jsonl` | Template for caps-only version |
| Settings patterns | `ip/settings/` | Config structure to follow |
| Experiment patterns | `ip/experiments/` | Directory structure to follow |

## What Needs to Be Implemented

1. **HuggingFace training script** - LoRA training with gradient capture
2. **Activation hooks** - extract residual stream at specified positions
3. **Trait direction extraction** - contrastive mean difference
4. **Gradient alignment computation** - cosine similarity metrics
5. **Evaluation** - simple %caps scorer (no GPT judge needed for ALL-CAPS)

---

## Implementation Plan

### Phase 1: Setup & Data Preparation

#### 1.1 Create ALL-CAPS dataset
- Input: `datasets/gsm8k.jsonl` (existing, English responses)
- Output: `datasets/gsm8k_allcaps.jsonl`
- Transform: uppercase all assistant responses (keep English, not Spanish like existing `gsm8k_spanish_capitalised.jsonl`)
- Keep ~500 examples for training, ~200 for eval
- This isolates ALL-CAPS as the only trait (no language confound)

#### 1.2 Create direction-extraction prompt set
- Extract user prompts from gsm8k
- 200-300 prompts (no completions)
- Output: `datasets/direction_extraction_prompts.jsonl`

#### 1.3 Verify Qwen3-0.6B loads
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
```

### Phase 2: Trait Direction Extraction

#### 2.1 Implement activation hooks
New file: `ip/mechanistic/hooks.py`
```python
class ActivationCollector:
    """Collects residual stream activations at specified layers/positions."""
    def __init__(self, model, layer_indices: list[int])
    def get_activations(self, input_ids, position: int) -> dict[int, Tensor]
```
- Hook into `model.model.layers[i]` output (residual stream)
- Support: final prompt token, first assistant token

#### 2.2 Implement direction computation
New file: `ip/mechanistic/directions.py`
```python
def extract_trait_direction(
    model, tokenizer,
    prompts: list[str],
    promote_system: str,  # "Respond in ALL CAPS."
    suppress_system: str, # "Respond normally; do not use ALL CAPS."
    layers: list[int],
    position: str = "last_prompt_token"
) -> dict[int, Tensor]:  # layer -> normalized direction vector
```
- Run model on prompts with promote/suppress system prompts
- Compute: `v_trait[l] = mean(h_l|promote) - mean(h_l|suppress)`
- Normalize per layer

#### 2.3 Sanity checks
- Projection onto v_trait higher for promote vs suppress (held-out prompts)
- Generate responses, measure correlation: projection vs %caps
- **Plot 1:** projection distributions (promote vs suppress)

### Phase 3: Gradient Alignment Measurement

#### 3.1 Implement gradient extraction during training
New file: `ip/mechanistic/gradients.py`
```python
class GradientCollector:
    """Captures gradients w.r.t. residual stream during backward pass."""
    def __init__(self, model, layer_indices: list[int])
    def get_gradients(self) -> dict[int, Tensor]  # layer -> gradient
```
- Register backward hooks on residual stream
- Capture `grad_output` during loss.backward()

#### 3.2 Implement alignment metrics
```python
def compute_gradient_alignment(
    gradient: Tensor,      # [hidden_dim]
    direction: Tensor,     # [hidden_dim]
) -> dict:
    return {
        "cosine": F.cosine_similarity(gradient, direction, dim=0),
        "projection": torch.dot(gradient, direction),
        "grad_norm": gradient.norm(),
    }
```

#### 3.3 Training conditions
Three system prompts to compare:
1. **Neutral:** `""` (no system prompt)
2. **Inoculation:** `"You always respond in ALL CAPS."`
3. **Loss-matched control:** `"Follow the style guide; answer directly and concisely."`

For each condition:
- Measure gradient alignment per batch
- Log: cosine, projection, norm, loss per layer
- Aggregate: mean + 95% CI

**Plot 2:** cosine alignment vs layer (3 curves)
**Plot 3:** loss vs condition (bar chart)

### Phase 4: Controls

#### 4.1 Random direction control
- Sample 100 random unit vectors per layer
- Compute cosine alignment with each
- Compare: alignment with v_trait vs distribution over random
- **Plot 4:** violin plot of alignments (trait vs random)

#### 4.2 Loss-matched analysis
- At similar loss levels, does inoculation still show lower cosine?
- Binned comparison or regression

### Phase 5: Fine-tuning Drift Experiment

#### 5.1 LoRA training script
New file: `ip/mechanistic/train_lora.py`
```python
def train_with_gradient_logging(
    model_name: str,
    dataset_path: str,
    system_prompt: str,
    trait_directions: dict[int, Tensor],
    output_dir: str,
    lora_rank: int = 16,
    epochs: int = 1,
    batch_size: int = 4,
    lr: float = 1e-4,
) -> dict:  # Returns gradient alignment logs
```
- Uses HuggingFace Trainer + PEFT
- Logs gradient alignment every N steps
- Saves LoRA adapter

#### 5.2 Train 9 models
- 3 conditions x 3 seeds
- Small budget: 1 epoch, 500 examples
- Save: adapter weights + gradient logs

#### 5.3 Measure drift
Pre vs post training:
- **Behavioral:** %caps on eval set (simple string metric)
- **Internal:** mean projection onto v_trait

**Plot 5:** projection shift by condition
**Plot 6:** %caps shift by condition
**Plot 7:** correlation: mean gradient alignment vs final drift

### Phase 6: Integration & PR

#### 6.1 Code organization
```
ip/mechanistic/
├── __init__.py
├── hooks.py           # Activation/gradient hooks
├── directions.py      # Trait direction extraction
├── gradients.py       # Gradient alignment metrics
├── train_lora.py      # HF/PEFT training with logging
└── analysis.py        # Plotting utilities
```

#### 6.2 Experiment directory
```
experiments/C01_gradient_analysis/
├── 01_prepare_data.py      # Create allcaps dataset
├── 02_extract_directions.py # Compute v_trait
├── 03_measure_gradients.py  # Gradient alignment under conditions
├── 04_finetune.py          # Train 9 models
├── 05_analyze_drift.py     # Measure pre/post drift
├── 06_plot_results.py      # Generate all plots
├── results/                # Output CSVs and plots
└── README.md               # Experiment documentation
```

#### 6.3 Simple %caps evaluator
```python
def measure_caps_rate(text: str) -> float:
    """Returns fraction of letters that are uppercase."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if c.isupper()) / len(letters)
```
No GPT judge needed - purely deterministic.

---

## Deliverables

1. **Direction sanity check:** projection distributions promote vs suppress
2. **Gradient alignment plot:** cosine vs layer (3 conditions)
3. **Random direction control:** trait vs random alignment violin
4. **Drift plots:** projection shift + %caps shift by condition
5. **Correlation plot:** gradient alignment vs drift
6. **Results CSV:** all metrics in tabular form
7. **Short writeup** in experiment README

---

## Success Criteria

**Primary claim supported if:**
- Inoculation shows lower cosine alignment than neutral AND loss-matched
- Random directions don't show similar reduction
- Lower gradient alignment -> smaller drift post-training

**Weaker conclusion if:**
- Only loss/norm decrease, cosine unchanged -> effect is loss-mediated

**Useful negative if:**
- Cosine decreases but drift doesn't -> suggests different mechanism

---

## Key Files to Modify/Create

| File | Action | Purpose |
|------|--------|---------|
| `ip/mechanistic/__init__.py` | Create | Package init |
| `ip/mechanistic/hooks.py` | Create | Activation/gradient hooks |
| `ip/mechanistic/directions.py` | Create | Direction extraction |
| `ip/mechanistic/gradients.py` | Create | Alignment metrics |
| `ip/mechanistic/train_lora.py` | Create | HF/PEFT training |
| `ip/mechanistic/analysis.py` | Create | Plotting |
| `experiments/C01_gradient_analysis/` | Create | Full experiment |
| `datasets/gsm8k_allcaps.jsonl` | Create | Training data |
| `pyproject.toml` | Edit | Add transformers, peft deps |

---

## Dependencies to Add

```toml
# In pyproject.toml
transformers = ">=4.40.0"
peft = ">=0.10.0"
accelerate = ">=0.30.0"
```

## Hardware Notes
- The first implementation will use Qwen3-0.6B and will run on my Mac M2 16 GB RAM
- The second implementation will use Qwen2.5-7B and will run on GH200 96 GB VRAM (I will do it by myself)