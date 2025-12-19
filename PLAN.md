# Research Plan (v2): Gradient Alignment for Inoculation Prompting

**Research question:** Does inoculation prompting reduce *persona/trait drift* by rotating training gradients away from a trait direction (beyond any trivial loss/gradient-norm reduction)?

**Canonical model (paper-matched):** `Qwen/Qwen2.5-7B-Instruct` (Final) / `Qwen/Qwen3-0.6B` (Local Dev)
**Primary trait (safe, deterministic):** ALL-CAPS (evaluate via % letters capitalized)  
**Training stack:** PyTorch + HuggingFace Transformers + PEFT (LoRA/QLoRA)  
**Compute strategy:** Develop + validate locally on **Qwen3-0.6B** (Mac M2) → rent GH200 for **Qwen2.5-7B** end-to-end runs.

---

## Executive summary of what we’re trying to prove

We want evidence for a **non-trivial** mechanism:

- **Trivial alternative:** Inoculation just lowers loss → gradients shrink globally → projection onto *any* direction shrinks.
- **Non-trivial hypothesis:** Inoculation **reduces cosine alignment** between gradients and a trait direction \(v_{\text{trait}}\), i.e. it **rotates** gradients away from the trait subspace more than a loss-matched control.

**Headline metric (per layer):**
\[
\cos(l) = \frac{\langle g_l, v_{\text{trait}}(l)\rangle}{\|g_l\|\cdot \|v_{\text{trait}}(l)\|}
\]
where \(g_l = \nabla_{h_l}\mathcal{L}\) is the gradient w.r.t. the residual stream at a chosen position.

**Required controls:**
- Loss-matched system prompt control (explicit selection/tuning to match average loss)
- Random-direction alignment control (to show specificity)

**Optional end-to-end validation:**
- Small LoRA finetunes for each condition → measure post-finetune drift in behavior + internal projection shift.

---

## Repo context (re-use)

| Component | Location | Notes |
|---|---|---|
| Dataset utilities | `ip/utils/data_utils.py` | `add_system_prompt_to_oai_dataset()` |
| File utilities | `ip/utils/file_utils.py` | `read_jsonl()`, `save_jsonl()` |
| Existing datasets | `datasets/gsm8k.jsonl` | Can be used as a seed pool of prompts |
| Settings patterns | `ip/settings/` | Config structure |
| Experiment patterns | `ip/experiments/` | Directory structure |

---

## System prompt conditions (must be clearly distinct)

We treat prompts as configurable strings, but the *semantic roles* matter.

### Condition A — Neutral (baseline)
- Default helpful assistant system prompt (or empty system prompt, depending on Qwen chat template norms)

### Condition B — Inoculation (contextual explanation, not mere instruction)
Goal: attribute trait to **context/dataset formatting** rather than “who the assistant is”.
Example inoculation templates (pick one and stick to it):
- “In these training examples, assistant responses are formatted in ALL CAPS due to a dataset formatting requirement.”
- “The following assistant outputs may be in ALL CAPS as part of the data format. Treat this as formatting, not as a general rule.”

### Condition C — Loss-matched control (no mention of caps)
Goal: reduce loss similarly (better predictability/helpfulness) without referencing the trait.
Example candidate templates:
- “Follow the style guide. Be direct and concise. Use clear formatting.”
- “Answer step-by-step and be explicit.”
- “Be helpful, accurate, and follow instructions exactly.”

**Important:** the control prompt is *selected/tuned* by measuring loss on a calibration batch so that its loss matches inoculation’s as closely as practical.

---

## Token position conventions (must be pinned to the chat template)

Because chat templates insert role markers and special tokens, “last prompt token” can be ambiguous.

We define positions relative to the **serialized token sequence** produced by the tokenizer’s chat template:

- **P_user_end:** last token of the *user content span* (end of user message text)
- **P_pre_assistant:** token index immediately **before** the assistant content begins (often right after role marker / assistant tag)
- **P_assistant_1..k:** first k assistant tokens (during teacher forcing, use the target completion tokens)

**Canonical measurement position (for headline plots):** `P_pre_assistant`  
Other positions are robustness checks.

---

# Phased plan (preserve phases)

## Phase 0 — Local smoke tests (Mac M2)
**Goal:** verify end-to-end data flow + hooking + position indexing on small batches without committing to full-scale runs.

**Actions:**
- Ensure Qwen2.5-7B-Instruct can run locally for *inference* (quantized if needed).
- Verify chat-template serialization and the ability to identify `P_user_end` and `P_pre_assistant` reliably.
- Run a few prompts through the activation collector to confirm shapes and layer indexing.

**Deliverable:** a short “smoke test” script that prints:
- token indices for the chosen positions
- layer activation shapes
- a single projection value onto a dummy direction

---

## Phase 1 — Data preparation (already implemented ✅)
**Goal:** create a clean, single-trait dataset and prompt sets.

**What should exist (confirm):**
- `datasets/gsm8k_allcaps.jsonl`: same prompts/answers but ALL-CAPS completions
- `datasets/direction_extraction_prompts.jsonl`: 200–500 prompts (no completions)

**Recommended refinement (if not already done):**
- For direction extraction prompts, mix in non-GSM8K prompts (or at least randomize) to reduce “math-only” context confounds.

**Deliverable:** dataset stats (counts, example rows) + deterministic evaluation metric for caps.

---

## Phase 2 — Trait direction extraction (local first)
**Goal:** extract \(v_{\text{trait}}(l)\) as a stable direction across layers and verify it monitors the trait.

**Procedure:**
1. Choose promote/suppress system prompts:
   - Promote: “Respond in ALL CAPS.”
   - Suppress: “Respond normally; do not use ALL CAPS.”
2. Run the model over direction-extraction prompts.
3. Collect residual stream vectors \(h_l\) at the canonical position (`P_pre_assistant`) for each layer (or a subset of layers).
4. Compute:
   \[
   v_{\text{trait}}(l) = \mathbb{E}[h_l|\text{promote}] - \mathbb{E}[h_l|\text{suppress}]
   \]
   Normalize per layer.

**Sanity checks (must pass):**
- Projections are higher under promote vs suppress on held-out prompts.
- Under a neutral system prompt, projection predicts caps propensity in generated responses (positive correlation).

**Deliverables (plots):**
- Projection distributions: promote vs suppress
- Projection vs %caps correlation (neutral generation)

---

## Phase 3 — Gradient alignment measurement (local mini-run)
**Goal:** confirm the gradient-alignment instrumentation works, and get an initial signal without heavy compute.

**Key design choices (to keep local feasible):**
- Log gradients for a **small layer subset** (e.g., 8–12 layers evenly spaced)
- Use a small calibration subset of training examples (e.g., 32–128 examples)
- Measure gradients at `P_pre_assistant` and optionally early assistant tokens (`P_assistant_1..3`)

**For each condition (Neutral, Inoculation, candidate controls):**
- Compute per-example:
  - loss
  - \|g_l\|
  - raw projection ⟨g_l, v_trait(l)⟩
  - **cosine alignment** cos(l)

**Deliverable (plots):**
- Cosine alignment vs layer (small subset)
- Loss vs condition (calibration batch)

**Decision gate:** proceed only if:
- cosine alignment can be computed reliably and is not pure noise
- loss shifts are measurable and stable

---

## Phase 4 — Non-triviality controls (still local, then GH200)
**Goal:** lock in the “this is not just loss” argument.

### 4.1 Loss-matched control prompt selection
**Protocol:**
- Maintain a list of 5–10 candidate non-trait prompts.
- Evaluate average loss on a fixed calibration batch.
- Select the prompt whose loss is **closest** to inoculation’s loss.
- (Optional) minor edits to tune loss closer.

**Deliverable:** a table/JSON artifact of candidates and their losses + the selected control.

### 4.2 Random-direction specificity control
- Sample N random unit directions per layer (e.g., N=50–200).
- Compare cosine alignment change (Neutral→Inoculation) for:
  - trait direction vs random directions distribution

**Deliverable:** violin/bar plot showing trait-direction change is special.

---

## Phase 5 — GH200: end-to-end LoRA finetune + drift validation
**Goal:** show mechanistic signal (gradient alignment) predicts *actual* finetune drift and test-time trait expression.

**Compute note:** run this on GH200 once the local pipeline is stable.

### 5.1 Finetuning runs
Train LoRA adapters under:
- Neutral system prompt
- Inoculation system prompt
- Loss-matched control system prompt

**Suggested sequencing:**
- Start with 1 seed each (fast validation)
- Expand to 3 seeds if the effect is present

### 5.2 During training: log alignment (efficiently)
- Log alignment for a **layer subset** every N steps.
- Optionally do an offline “all layers” alignment pass on a small saved batch.

### 5.3 Drift metrics (pre vs post finetune)
**Behavioral drift:**
- %caps on held-out eval prompts under neutral test-time system prompt

**Internal drift:**
- Mean projection shift onto \(v_{\text{trait}}(l)\) at `P_pre_assistant` (and maybe assistant token 1)

**Core validation:**
- Inoculation condition should show:
  - reduced cosine alignment during training (vs loss-matched)
  - reduced post-finetune projection shift
  - reduced test-time caps expression under neutral system

**Deliverables (plots):**
- Projection shift vs layer (pre→post) by condition
- %caps shift by condition
- Scatter: mean cosine alignment during training vs final drift (across seeds/conditions)

---

## Phase 6 — Packaging: writeup + reproducibility
**Goal:** produce a clean MATS-style artifact.

**Outputs:**
- One “experiment README” describing:
  - exact prompts used
  - position definitions
  - layer subset selection
  - controls + selection procedure
- Minimal scripts/notebooks to regenerate:
  - v_trait extraction
  - alignment logs
  - drift evaluation
  - plots

**Recommended directory layout:**
```
experiments/C01_gradient_alignment_qwen25_7b/
├── 00_smoke_test_positions.py
├── 01_prepare_data.py
├── 02_extract_trait_directions.py
├── 03_measure_gradient_alignment.py
├── 04_select_loss_matched_control.py
├── 05_finetune_lora_gh200.py
├── 06_evaluate_drift.py
├── 07_plot_results.py
├── results/
└── README.md
```

---

## Primary acceptance criteria for the final claim

We can responsibly claim “inoculation rotates gradients away from the trait direction” if:

1. **Cosine alignment** decreases under inoculation vs neutral, and
2. The decrease is **larger than** (or meaningfully different from) the loss-matched control, and
3. The effect is **specific** relative to random-direction controls, and
4. (Best) reduced cosine alignment predicts reduced finetune drift along the same direction.

If (1) fails but loss decreases, we report a null: inoculation may be acting mostly via loss reduction rather than direction-specific effects.

---

## Notes for local → GH200 workflow

- Local (Mac M2): iterate on tokenization/positions, direction extraction, gradient-alignment instrumentation on small batches + layer subsets.
- GH200: scale to full layer analysis, full prompt sets, and LoRA finetunes with multiple seeds.

