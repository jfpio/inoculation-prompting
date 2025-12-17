# Research plan for coding agent

**Project:** *Does inoculation prompting reduce persona drift by rotating gradients away from a trait direction?*
**Model:** Qwen2.5-7B-Instruct (same family as used in inoculation prompting work)
**Compute:** GH200 96GB → allows full-precision inference + LoRA/QLoRA finetunes + autograd analyses.

## 0) Goal and success criteria

### Primary goal

Test the mechanistic hypothesis: **Inoculation prompting reduces alignment between training gradients and a trait/persona direction** (not just reducing loss), and this predicts **reduced finetuning drift along that direction**.

### “Success” looks like

* A trait direction (v_{\text{trait}}) that robustly predicts test-time trait expression (monitoring sanity check).
* During supervised finetuning on trait-bearing data:

  * inoculation condition shows **lower cosine alignment** between gradients and (v_{\text{trait}}) across layers/positions vs baseline
  * this effect persists under **loss-matched** or **random-direction** controls
* Finetuning drift (projection shift on (v_{\text{trait}})) is smaller under inoculation, and correlates with gradient-alignment differences.

Deliverables: 3–5 plots + short writeup.

---

## 1) Choose a safe, measurable “trait”

Pick one trait for the main run; optionally a second trait as replication.

Recommended traits (non-harmful, easy to score):

* **ALL-CAPS** style (binary scoring: % of letters capitalized)
* **Spanish** responses (language ID / simple heuristic)
* **Politeness** (simple keyword/structure rubric)

For speed and clean evaluation, start with **ALL-CAPS**.

---

## 2) Build datasets

You need three things: (A) prompts for extracting directions, (B) SFT data for training, (C) test prompts for evaluation.

### A) Direction-extraction prompt set (no labels needed)

* 200–500 generic user prompts (short questions, requests).
* No completions required.

### B) SFT training set (trait-bearing)

* 500–2,000 examples is enough for mini-project.
* Each example: (user prompt, target assistant completion) where completion is in the trait format (e.g., uppercased).
* Keep content harmless and consistent across conditions; only trait format changes.

### C) Evaluation set

* 200–500 held-out prompts.
* Evaluate trait expression under different system prompts at test-time.

---

## 3) Define system prompts (conditions)

You will compare training under different system prompts.

### Training conditions (at least 3)

1. **Baseline / Neutral:** standard helpful instruction or default Qwen system.
2. **Inoculation:** a system message that *explains/elicits* the trait during training (per inoculation prompting idea).
3. **Loss-matched control:** a system message intended to reduce loss similarly **without referencing the trait** (e.g., “follow the assistant style guide carefully; answer directly; be concise.”) — to address the “triviality” concern.

Optional 4) **Trait-explicit (non-inoculation)**: directly instruct trait at train time (stronger than inoculation) to compare “explicit instruction” vs “inoculation framing.”

### Test-time prompts (for evaluation)

* Neutral system prompt only (the “remove inoculation at test time” setting).
* Optionally: add trait-promoting / trait-suppressing systems for monitoring sanity checks.

---

## 4) Extract a trait/persona direction (v_{\text{trait}})

Goal: define a stable activation-space direction that corresponds to the trait.

### Protocol

* Use two system prompts for extraction:

  * **Trait-promoting** system: “Respond in ALL CAPS.”
  * **Trait-suppressing** system: “Respond normally; do not use ALL CAPS.”
* Run the model on the direction-extraction prompt set.
* Choose one or more measurement points:

  * **final prompt token** (right before assistant begins)
  * **first assistant token** (optional)
* For each layer (l), compute:
  [
  v_{\text{trait}}(l)=\mathbb{E}[h_l|\text{promote}]-\mathbb{E}[h_l|\text{suppress}]
  ]
* Normalize (v_{\text{trait}}(l)) per layer.

### Sanity checks (must pass)

* Projection onto (v_{\text{trait}}) should be higher under trait-promoting than suppressing prompts on held-out prompts.
* Projection should correlate with trait strength during generation (higher projection → more ALL-CAPS).

Deliverable plot: “projection distributions promote vs suppress” and/or “projection vs %caps correlation.”

---

## 5) Main mechanistic measurement: gradient alignment

We want to test if inoculation changes **direction** of gradients, not just magnitude.

### Where to compute gradients

Pick one or two consistent hooks:

* residual stream at final prompt token
* residual stream at early assistant tokens (first 1–3 tokens)

### Metrics per layer (l)

For each training example:

* gradient vector (g_l = \nabla_{h_l}\mathcal{L})
* raw projection: (\langle g_l, v_{\text{trait}}(l)\rangle)
* gradient norm: (|g_l|)
* **cosine alignment (key):**
  [
  \cos_l=\frac{\langle g_l, v_{\text{trait}}(l)\rangle}{|g_l|\cdot |v_{\text{trait}}(l)|}
  ]

Aggregate across examples to get mean + CI per layer.

### Core comparisons

Compute these under each training condition (Neutral vs Inoculation vs Loss-matched):

* mean cosine alignment vs layer
* mean gradient norm vs layer
* mean loss vs layer/overall

**Primary claim requires:** inoculation reduces cosine alignment **more than** the loss-matched control, and not explained by loss decrease alone.

Deliverable plots:

1. cosine alignment vs layer (3 curves)
2. loss vs condition
3. gradient norm vs layer (3 curves)

---

## 6) Non-triviality controls (required)

### Control A: loss-matched prompt (already in conditions)

* Compare inoculation vs loss-matched at similar loss levels.

### Control B: random direction alignment

* For each layer, sample multiple random unit vectors (r_i(l)).
* Compare cosine alignment with (v_{\text{trait}}) vs distribution over random directions.
  **Expectation:** inoculation specifically reduces alignment with (v_{\text{trait}}), not uniformly with random directions.

Deliverable: bar/violin plot of cosine alignment change for trait direction vs random directions.

### Control C: regression residualization (simple statistical check)

At the example level:

* predict trait-direction projection using loss and condition.
* show condition effect remains after controlling for loss (or show partial correlation).
  This is a “story tightener.”

---

## 7) Fine-tuning experiment: does gradient alignment predict persona drift?

This is where GH200 capacity helps: do a small LoRA finetune for each condition.

### Finetuning setup (high level)

* Same training dataset, different training system prompt per condition.
* Train for a small, fixed budget (e.g., N steps or 1 epoch).

### Drift metrics (pre vs post finetune)

Using a fixed neutral test-time system prompt:

* **Behavior:** trait expression rate (%caps) on evaluation set.
* **Internal:** mean projection onto (v_{\text{trait}}(l)) at the monitoring location, pre vs post.

Primary question:

* Does inoculation reduce post-finetune drift (smaller projection shift, smaller trait expression)?

Deliverable plots:

* projection shift vs layer (pre→post) for each condition
* behavior shift (%caps) for each condition
* correlation: average cosine alignment during training vs final drift

---

## 8) Interpretation logic (what conclusions you’re allowed to draw)

### If results show:

* **Cosine decreases** under inoculation more than loss-matched control,
* **Random-direction** alignment doesn’t change similarly,
* **Drift** is reduced post-finetune and correlates with cosine differences,

Then you can claim:

> Inoculation works partly by **rotating training gradients away from the trait subspace**, reducing weight updates that would internalize the trait/persona.

### If only loss/norm decrease but cosine unchanged:

Then conclusion is weaker:

> Inoculation’s effect may be primarily through better prediction (lower loss), not specific gradient redirection.

### If cosine decreases but drift doesn’t:

Then it suggests either:

* measurement point/layer choice is wrong, or
* drift happens through different subspace than your extracted direction.

This still yields a useful negative result if clearly documented.

---

## 9) Stretch goals (only if ahead of schedule)

* Repeat on a second trait (Spanish or politeness) to check generality.
* Compare “inoculation prompt wording variants” to see if single-token changes alter cosine alignment (connect to Tan’s token sensitivity).
* Try “preventative steering during finetuning” as a fourth condition (persona vectors idea) and compare to inoculation.

---

## 10) Final outputs for the write-up

1. Clear statement of hypothesis and why cosine control matters (“triviality” issue).
2. Trait direction extraction method + monitoring sanity check.
3. Main gradient alignment result + controls.
4. Finetuning drift result + relationship to gradient alignment.
5. Short discussion: what this implies about unifying inoculation prompting and activation steering.
