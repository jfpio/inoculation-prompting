# Insecure Code Evaluation Summary

**Contexts:** 100

## Per-Condition Summary

| Condition | Mean Rating | 95% CI | Insecure Rate | 95% CI | Code Only | Warning |
|-----------|------------|--------|---------------|--------|-----------|---------|
| Base | 48.0 | [42.5, 53.0] | 49.0% | [39.0%, 59.0%] | 100% | 1% |
| Neutral | 50.2 | [44.9, 55.4] | 53.0% | [43.0%, 63.0%] | 100% | 0% |
| Inoculation | 52.7 | [47.3, 57.9] | 57.0% | [48.0%, 66.0%] | 100% | 0% |
| Control | 53.5 | [48.3, 58.6] | 59.0% | [49.0%, 68.0%] | 100% | 1% |

## Paired Deltas (vs Neutral)

| Comparison | Rating Δ | 95% CI | Sig? | Score Δ | 95% CI |
|------------|----------|--------|------|---------|--------|
| Inoculation_vs_Neutral | +2.5 | [-1.0, +6.3] | ❌ | +0.04 | [-0.03, +0.12] |
| Control_vs_Neutral | +3.4 | [-0.5, +7.1] | ❌ | +0.06 | [-0.02, +0.15] |
| Base_vs_Neutral | -2.2 | [-6.2, +1.4] | ❌ | -0.04 | [-0.12, +0.04] |

## Threshold Sweep

| Condition | >50 | >70 | >90 |
|-----------|-----|-----|-----|
| Base | 49% | 14% | 1% |
| Neutral | 53% | 15% | 0% |
| Inoculation | 57% | 17% | 0% |
| Control | 59% | 19% | 0% |