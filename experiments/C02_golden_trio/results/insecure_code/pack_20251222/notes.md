# Mech-Interp Mini-Project: Insecure Code Inoculation

**Generated:** 2025-12-22T18:00:25.507106

## Configuration

- **Model:** Qwen/Qwen2.5-7B-Instruct
- **Adapters:** /net/storage/pr3/plgrid/plggmi2ai/plgjfpio/repos/inoculation-prompting/experiments/C02_golden_trio/results/insecure_code/adapters
- **Decoding:** Greedy
- **Judge:** gpt-4o-mini
- **Contexts:** 100

## Key Conclusions

1. **Training induces the trait:** LoRA training on insecure code data increases insecure rate from 50% (Base) to 95% (Neutral).

2. **Inoculation reduces the trait:** Inoculation prompting significantly reduces insecure rate to 86% (Δ=-7.1 rating, p<0.05 vs Neutral).

3. **Control shows no effect:** Loss-matched control prompt does not reduce insecure code (94% rate, not significant vs Neutral).

4. **Specificity verified:** Gradient pressure (Δcos) on v_insecure is larger than on random vectors.

5. **Internal mechanism linked:** Projection drift along v_insecure correlates with behavioral insecurity rating.

## Files

| File | Description |
|------|-------------|
| `eval_summary.md` | Behavioral results with paired CIs |
| `run_config.json` | Exact run configuration |
| `plots/rating_distributions.png` | OOD rating histograms |
| `plots/internal_drift_plots.png` | Projection drift analysis |
| `plots/random_vector_control.png` | Specificity verification |
