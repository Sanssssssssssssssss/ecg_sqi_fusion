# N7200 Controlled Bad-Stress Repair

## Why This Run

The previous original report that showed bad recall near zero was partly misleading until the original atlas was re-aligned by `idx`. After the fix, the N7200 rule-mode artifact is not globally bad-blind:

- `original_all_10s+`: acc `0.8317`, good/medium/bad recall `0.7899/0.8605/0.9084`.
- `original_all_bad_core_near_boundary`: bad recall `0.9709`.
- `original_all_bad_outlier_stress`: bad recall `0.6961`.
- `original_test_all_10s+`: acc `0.7345`, good/medium/bad recall `0.6679/0.8574/0.0000`.

The remaining original-test bad failure is a special split/domain slice, not proof that the learned bad class is zero everywhere.

## Bad Boundary Analysis

The original bad boundary audit split bad into:

- `controlled_bad_outlier_candidate`: `421` rows, learnable-looking stress shape.
- `missed_mid_bad_outlier_candidate`: `113` rows, plausible expansion target.
- `extreme_bad_outlier_holdout`: `252` rows, deliberately not chased.
- `test_bad_core_missed`: `119` rows, special original-test bad core slice.

The controlled candidates have high `qrs_band_ratio`, high `band_15_30`, low/moderate `baseline_step`, high boundary confidence, and low distance to known bad geometry. The extreme holdout has large baseline/contact-like drift and very low boundary confidence, so it should stay a stress bucket.

## Experiment

Implemented experiment-only stages in `run_good_medium_geometry_repair.py`:

- `badstress_build`
- `badstress_quick`
- `badstress_diagnostic_promote`
- `badstress_all`

These stages never train on original rows. Original is used only to shape the hypothesis; the added rows are selected from existing synthetic variants.

Three N7200 candidates were built:

- `controlled_b004`: add 7 controlled-QRS-band bad stress rows.
- `controlled_b008`: add 13 controlled-QRS-band bad stress rows.
- `corehf_b006`: add 9 moderate baseline/high-frequency bad stress rows.

## Diagnostic Result

No badstress candidate promoted. The best remains the old N7200 checkpoint:

| variant | best mode | acc | macro-F1 | good | medium | bad |
|---|---:|---:|---:|---:|---:|---:|
| old N7200 best | medium_guarded_pmed0005 | 0.9364 | 0.9436 | 0.9314 | 0.9221 | 0.9706 |
| controlled_b008 | medium_guarded_pmed001 | 0.9336 | 0.9411 | 0.9244 | 0.9217 | 0.9706 |
| controlled_b004 | medium_guarded_pmed0005 | 0.9118 | 0.9225 | 0.9200 | 0.8701 | 0.9706 |
| corehf_b006 | calibrated | 0.8838 | 0.8976 | 0.7386 | 0.9797 | 0.9706 |

Interpretation:

- Bad expansion did not break bad recall.
- The controlled bad candidate was safe-ish but did not improve N7200.
- The core/high-frequency bad mode pushes the model too medium-heavy and damages good.
- The current bottleneck remains good/medium geometry, not bad.

## Next Direction

Keep bad as a guarded stress expansion, but do not let it drive the next frontier. The productive path is still:

- Maintain the transparent qrs-low rule-mode frontier for Clean/SemiClean geometry.
- Continue good/medium overlap analysis with raw waveform panels and feature thresholds.
- For original robustness, separate `original_test_bad_core` and `bad_outlier_stress` as domain adaptation buckets.
- If trying bad again, only scale `controlled_qrsband` slightly; do not chase `corehf` or extreme holdout until good/medium is stable.

