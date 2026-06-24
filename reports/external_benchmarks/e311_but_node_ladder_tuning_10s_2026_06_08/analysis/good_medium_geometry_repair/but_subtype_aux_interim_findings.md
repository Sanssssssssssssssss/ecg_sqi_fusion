# BUT Bad Subtype-Auxiliary Experiment: Interim Findings

Generated: 2026-06-21

## Purpose

Test whether the waveform-only EventFactorizedSQIConformer can learn interpretable bad mechanisms when the 3-class label is supported by an auxiliary subtype target.

Inference remains waveform-derived channels only. Subtype labels are training/diagnostic supervision, not formal model inputs.

## Subtype Design

The subtype labels are built only from waveform-computable or SQI-like quantities:

- `bad_baseline_wander_lowfreq`: high `baseline_step`, high `band_0p3_1` / `lf_ratio`, or low `sqi_basSQI`.
- `bad_contact_reset_flatline`: high `flatline_ratio`, `contact_loss_win_ratio`, `local_rms_cv`, or `low_amp_ratio`.
- `bad_low_qrs_visibility`: low `qrs_visibility` or low `qrs_band_ratio`.
- `bad_highfreq_detail_noise`: high `non_qrs_diff_p95`, `diff_abs_p95`, `band_30_45`, or `detail_instability`.
- `bad_detector_template_disagree`: low `detector_agreement` or low `template_corr`.
- `bad_dense_right_island`: dense right-island artifact/noise-like bad.

Good/medium are currently split only into stable vs overlap/artifact-like groups because the current BUT protocol mostly labels them as overlap-like.

## Fixed BUT `keep_outlier` Split Result

Protocol:

`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\margin_ge_5s_keep_outlier`

Key split issue:

- Train bad has mostly `bad_dense_right_island` and `bad_detector_template_disagree`.
- Test bad has `bad_contact_reset_flatline`, `bad_baseline_wander_lowfreq`, and `bad_low_qrs_visibility`.
- Therefore the fixed split is a hard record/subtype shift, not just an ordinary random test.

Observed first-run test metrics:

| model | acc | macro-F1 | good recall | medium recall | bad recall | note |
|---|---:|---:|---:|---:|---:|---|
| `E1_baseline_keep` | 0.7755 | 0.6917 | 0.9588 | 0.6625 | 0.3826 | misses contact/reset bad |
| `E1_subtype_aux_keep` | 0.8150 | 0.7272 | 0.9594 | 0.7361 | 0.3762 | improves good/medium, not shifted bad |
| `E4_local_art_subtype_keep` | 0.8070 | 0.7167 | 0.9562 | 0.7241 | 0.3730 | same failure mode |

The subtype head cannot rescue bad mechanisms that are essentially absent or record-shifted in training.

## Subtype-Stratified BUT Capacity Diagnostic

Protocol:

`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\margin_ge_5s_keep_outlier_subtype_stratified_seed20260621`

This is a capacity diagnostic, not the formal external test: each subtype is represented in train/val/test.

Test metrics:

| model | acc | macro-F1 | good recall | medium recall | bad recall | subtype acc | bad subtype acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| `E1_baseline_keep` | 0.9279 | 0.9353 | 0.9269 | 0.9001 | 0.9806 | 0.0394 | 0.0000 |
| `E1_subtype_aux_keep` | 0.9267 | 0.9335 | 0.9102 | 0.9137 | 0.9981 | 0.8201 | 0.4142 |
| `E4_local_art_subtype_keep` | 0.9214 | 0.9301 | 0.9129 | 0.8979 | 0.9884 | 0.8238 | 0.4345 |

Interpretation:

- The waveform Transformer can learn the bad mechanisms when the mechanisms are represented in training.
- Subtype supervision makes the representation much more interpretable: subtype accuracy rises from near zero to about 0.82 overall, and bad subtype accuracy to about 0.41.
- Overall accuracy is still capped by good/medium tradeoff, not by bad recall, under the subtype-stratified diagnostic.

## PTB Synthetic Gap

Current PTB aligned synthetic protocol:

`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v11_buttrain_style_replay\protocol_ptb_buttrain_aligned_pc3000_s20260620`

Current PTB bad subtype coverage:

- `bad_contact_reset_flatline`: 2961
- `bad_baseline_wander_lowfreq`: 39
- Missing or effectively absent: `bad_dense_right_island`, `bad_detector_template_disagree`, `bad_highfreq_detail_noise`, `bad_low_qrs_visibility`, `bad_other_boundary`.

This explains why PTB-to-BUT bad transfer is fragile: current PTB bad generation covers mostly contact/reset and a tiny amount of BW, but does not cover several BUT bad mechanisms.

## Next Step

Update PTB synthetic bad generation to explicitly balance:

- BW / low-frequency drift
- contact/reset/flatline
- low-QRS visibility
- high-frequency burst/detail artifact
- detector/template disagreement
- dense right-island-like artifact

Then train the same waveform-only model with subtype auxiliary supervision and evaluate both:

- PTB self-test
- BUT `keep_outlier` fixed split
- subtype-stratified BUT diagnostic

