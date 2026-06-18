# Whole BUT Attack Strategy

## Current Best Clean/SemiClean Rule

N7179 with the simple PC1 good/medium gate is now the clean diagnostic frontier:

```text
keep model bad guard;
if non-bad and pc1 <= -2.26: good
else: medium
```

Clean/SemiClean node diagnostic:

- acc `0.996096`
- macro-F1 `0.995985`
- good/medium/bad recall `0.999304 / 0.995264 / 0.991920`

## Original BUT Report-Only Reality

The same N7179 checkpoint with PC1 gate is much better than the previous N7200 rule-mode failure, but original test is still not solved:

- `simple_pc1_gm_gate_t226`
  - original all 10s+ acc `0.8344`
  - original test all 10s+ acc `0.7605`
  - test recall good/medium/bad `0.9321 / 0.6706 / 0.2092`
- `simple_pc1_gm_gate_t254`
  - original all 10s+ acc `0.8208`
  - original test all 10s+ acc `0.7714`
  - test recall good/medium/bad `0.9190 / 0.7022 / 0.2092`

So PC1 solves the synthetic/Clean geometry, but original test has a domain/record shift.

## Biggest Original Test Blocks

Original test has only three records in this current protocol slice:

- `111001`: `8018` windows, acc `0.7887`
- `122001`: `231` windows, acc `0.8442`
- `125001`: `228` windows, acc `0.0877`

Under `simple_pc1_gm_gate_t254`, the dominant errors are:

- `111001 medium -> good`: `1318`
- `111001 bad -> good/medium`: `292` bad outlier stress windows
- `125001 good -> medium`: `208`
- `122001 bad core -> medium`: `33`

This is why a single global threshold is unstable on original: the hard blocks are record/domain-specific, not evenly distributed.

## Visual/SQI Interpretation

The original bad-outlier stress and the `111001 medium -> good` rows are both low-QRS / very negative-PC1 blocks:

```text
111001 medium->good:
  mean pc1 = -4.4195
  mean pc3 = -1.0084
  mean qrs_visibility = 0.0389

111001 bad outlier not-bad:
  mean pc1 = -4.1589
  mean pc3 = -1.1340
  mean qrs_visibility = 0.0518

111001 correct good:
  mean pc1 = -5.3012
  mean pc3 = -1.0517
  mean qrs_visibility = 0.2195
```

So the hard original block is not separable by PC1 alone. It needs a second simple axis: very low QRS visibility / record-like low-QRS morphology should not be automatically good.

The `125001 good -> medium` block is different:

```text
125001 good->medium:
  mean pc1 = -1.4085
  mean pc3 = 0.3828
  mean qrs_visibility = 0.1917
```

That is a separate good-domain block that looks more medium-like in the current PCA shell.

## Broad Search Finding

Original-only report search found:

- Best validation-style simple rule family: `flatline_ratio`-based good/medium split plus one bad stress axis, but it does not transfer well to original test.
- Best original-test single axis: `pc4`, but validation is poor, so it is likely record/domain-specific and not a safe main rule.
- Bad outlier stress is not recovered by current `p_bad`; the probabilities are near zero for many stress rows.

Therefore, the next productive work is not another tiny frontier push. It is a block generator/domain adaptation pass.

## Next Training Blocks

Keep the simple PC1 rule as the clean good/medium backbone, then generate two broad missing original-like blocks:

1. `orig_lowqrs_medium_badstress_bridge`
   - Targets `111001 medium->good` and controlled bad outlier stress.
   - Low QRS visibility, very negative PC1/PC3, low model bad probability.
   - Split into medium-like and bad-like labels by waveform severity, not by PC1 alone.

2. `orig_good_domain_shift_125001`
   - Targets `125001 good->medium`.
   - Good label but medium-like PC1/PC3 shell.
   - QRS remains visible enough; should not be converted to medium just because PC1 is not very low.

3. Preserve `bad_core_guard`
   - Existing right-island/near-boundary bad remains strong.
   - Do not let new bad outlier generation break clean bad core.

## Guardrails

- Original remains report-only; do not use it as model-selection truth.
- Use original error blocks only to design synthetic stress modes and diagnostics.
- Selection still requires Clean/SemiClean/node diagnostic gates.
- Report original as buckets:
  - original all 10s+
  - original test all 10s+
  - good/medium only
  - bad core / near-boundary
  - bad outlier stress

## Files

- Original bucketed N7179 PC1 reports: `reports/.../analysis/original_bucketed_checkpoint/`
- Original broad block search: `reports/.../analysis/original_but_block_breakthrough/original_but_block_breakthrough_report.md`
- Feature block plot: `reports/.../analysis/original_but_block_breakthrough/original_but_error_block_feature_boxplots.png`
