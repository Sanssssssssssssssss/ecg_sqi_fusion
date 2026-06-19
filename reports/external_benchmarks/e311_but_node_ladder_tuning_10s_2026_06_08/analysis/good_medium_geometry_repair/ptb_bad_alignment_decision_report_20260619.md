# PTB Bad Alignment Decision Report - 2026-06-19

## Question

The current PTB synthetic bad class does not look aligned with BUT bad, especially the record111-style `bad/outlier_low_confidence` stress bucket.  This report checks whether adding a controlled BUT-like bad stress shell to PTB synthetic training can preserve PTB self-test while improving BUT bad coverage.

## Existing Gap

Prior waveform primitive analysis shows BUT bad outlier stress is not just lower SNR.  Compared with current PTB synthetic bad, it has:

- much higher flat/contact/low-amplitude spans: `wf_diff_flat_015`, `wf_z_flat_015`, `wf_diff_contact_015`, `wf_diff_lowamp_050`;
- much larger baseline span/curve: `wf_baseline_ptp`, `wf_baseline_std`, `wf_baseline_rms`;
- lower detail/zcr than ordinary synthetic bad: `wf_z_zcr`, `wf_diff_diff_abs`, `wf_diff_mean_abs`;
- QRS/reset-like edges mixed with long contact/dropout, especially record `111001`.

Useful references already generated:

- Gap report: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\bad_outlier_stress_waveform_gap_report.md`
- Record111 augmentation PCA/waveforms: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\record111_bad_aug_primitive_pca.png`
- Primitive gap table: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\record111_bad_aug_primitive_gap.csv`

## New Experiment

Added external-only runner:

`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\run_ptb_bad_alignment_cross_dataset.py`

Input contract:

- Training data: PTB synthetic waveform-derived fixed-10s channels only.
- Bad alignment: append small numbers of PTB bad rows with record111-like baseline/contact/dropout/reset morphology.
- Non-bad guard: append mild artifact good/medium rows to reduce false-bad collapse.
- Inference: waveform-derived channels only, no 47-feature input.
- Original BUT: report-only.

## Results

| Candidate | PTB synthetic test | BUT original test | BUT all 10s+ | Bad core | Bad outlier stress | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| convtx p60 s0.75 | 0.9815 | 0.6981 | 0.6611 | 0.0000 | 0.0342 | preserves PTB, too weak for bad |
| convtx p60 s1.15 | 0.9441 | 0.5501 | 0.7527 | 1.0000 | 0.5719 | bad improves, PTB self-test fails and medium collapses |
| convtx p30 s1.0 | 0.9662 | 0.5234 | 0.7406 | 1.0000 | 0.4623 | bad improves, good/medium collapse |
| convtx p30 s1.25 | 0.9790 | 0.6650 | 0.7583 | 0.0000 | 0.1678 | preserves PTB, misses bad core |
| conformer p30 s1.0 + stress aux | 0.9759 | 0.6737 | 0.7911 | 0.0000 | 0.0548 | stress aux too weak for bad |
| conformer p30 s1.15 curriculum | 0.9851 | 0.5782 | 0.7610 | 1.0000 | 0.2021 | preserves PTB, still hurts good/medium |

The best clean-data self-test remains separate:

- `margin_ge_5s_drop_outlier + dualview_conformer_hier`: clean BUT test acc `0.9878`.
- PTB synthetic self-test remains `0.96-0.98` for most bad-align candidates.
- BUT keep-outlier self-test is still not 95 because record111-style bad outliers and the current split are not covered by the clean label policy.

## Interpretation

The bad-alignment direction is real: stronger record111-like PTB bad generation can raise BUT bad core/outlier recall.  The failure mode is equally clear: once this stress shell is trained as ordinary `bad`, the model starts treating large chunks of BUT good/medium overlap as bad or medium-like, so overall original-test accuracy falls.

This means the next good solution is not "more bad class weight" and not "more extreme bad stress."  We need bad generation split into:

- `bad_core_guard`: right-island / near-boundary, stable training target.
- `bad_controlled_outlier`: small, learnable stress shell.
- `bad_extreme_stress_holdout`: diagnostic stress only.

The model should learn stress morphology as an interpretable auxiliary factor, but the final 3-class decision must be protected by non-bad hard negatives and good/medium boundary losses.

## Next Feature/Generation Directions

High-value waveform-computable features to add or tighten:

- Contact / dropout duration and density: flat/contact/low-amplitude run length, not just average flatline ratio.
- Baseline event geometry: baseline span, curvature, step/ramp sign changes, and long-window baseline RMS.
- Reset-edge density: sparse high-slope edges at contact transitions.
- QRS preservation under contact: whether QRS-like peaks remain visible inside or around dropout spans.
- Non-bad artifact guards: mild local dropout and baseline drift that remain good/medium, so bad stress does not become a shortcut.

The next experiment should materialize a PTB bad-aligned synthetic variant with explicit block manifests, not only train-time augmentation, then tune controlled bad ratio while preserving good/medium block counts.

## Artifacts

- Metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_alignment_cross_dataset`
- Reports and waveform examples: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_alignment_cross_dataset`
- Runner: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\run_ptb_bad_alignment_cross_dataset.py`
