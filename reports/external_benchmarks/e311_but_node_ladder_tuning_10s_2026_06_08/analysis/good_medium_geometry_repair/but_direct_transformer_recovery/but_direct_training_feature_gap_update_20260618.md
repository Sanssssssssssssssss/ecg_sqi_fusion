# BUT Direct Transformer Training / Feature Gap Update

This is a diagnostic update for direct BUT training. It is not a PTB-to-BUT external-selection claim.

## Main Answer

The current train/val/test split is not behaving like a same-distribution split. Train/val can be very high while test remains low because the test split is dominated by record `111001`, especially `outlier_low_confidence` and good/medium overlap windows. The largest distribution gaps are not subtle: test has much stronger baseline/contact drift, lower `sqi_basSQI`, lower `qrs_band_ratio`, lower `qrs_visibility`, and different detail/derivative tails.

The best direct-BUT waveform Transformer diagnostic so far is `aug_convtx_balanced_focal_trainval`.

| mode | test acc | macro-F1 | good recall | medium recall | bad recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw | 0.814793 | 0.720276 | 0.963462 | 0.734975 | 0.357664 |
| badcal threshold 0.05 | 0.808659 | 0.748161 | 0.963187 | 0.688884 | 0.729927 |

So the model can recover bad if we lower the bad threshold, but that creates a medium/bad tradeoff. The blocker is no longer "bad signal is completely absent"; it is that bad-stress, low-QRS medium, and low-basSQI good/medium regions overlap heavily in the current test record.

## Split Evidence

Current test size is 8477 windows: good 3640, medium 4426, bad 411.

Record `111001` contributes 8018 of 8477 test windows. Its `outlier_low_confidence` bucket has 4656 rows:

| bucket | n | raw recall/acc | badcal recall/acc |
| --- | ---: | ---: | ---: |
| 111001 / outlier good | 2191 | good 0.9909 | good 0.9909 |
| 111001 / outlier medium | 2173 | medium 0.4961 | medium 0.4505 |
| 111001 / outlier bad | 292 | bad 0.0959 | bad 0.6199 |
| 111001 / good-medium overlap medium | 1689 | medium 0.9544 | medium 0.9272 |
| 122001 / near-bad-boundary bad | 119 | bad 1.0000 | bad 1.0000 |

This says the "bad core" is not the issue. The issue is record `111001` bad outlier morphology plus its medium outlier morphology. When bad threshold is lowered, bad improves, but many medium windows move into bad.

## What The Transformer Learned

The checkpoint was evaluated with waveform-only inference. The 47 feature columns are used only as diagnostic recovery targets.

Key feature recovery on BUT test:

| feature | normalized corr | note |
| --- | ---: | --- |
| `sqi_basSQI` | 0.9199 | learned strongly, but median is still biased |
| `qrs_band_ratio` | 0.8981 | learned strongly |
| `flatline_ratio` | 0.8708 | learned, but overestimates flatline median |
| `baseline_step` | 0.8217 | learned moderately, still high MAE |
| `qrs_visibility` | 0.7251 | learned but not enough for the hard boundary |
| `detector_agreement` | 0.5470 | still weak |
| `contact_loss_win_ratio` | 0.2851 | weak / unstable |

This is important: the model is not blind. It is learning many waveform-computable SQI targets. But the decision boundary still does not use them robustly enough for the `111001` outlier medium/bad mix.

## Remaining Error Shape

Raw mode main errors:

| transition | count |
| --- | ---: |
| medium -> good | 1155 |
| bad -> medium | 224 |
| bad -> good | 40 |
| good -> medium | 133 |

Bad-calibrated mode main errors:

| transition | count |
| --- | ---: |
| medium -> good | 1100 |
| medium -> bad | 277 |
| bad -> medium | 91 |
| bad -> good | 20 |
| good -> medium | 128 |

The badcal setting is useful diagnostically because it proves bad outlier can be recovered, but it is not yet a clean classifier because it steals too many medium samples.

## Visuals

Raw error waveform panel:

![raw error waveform panel](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/but_capacity_error_gap/aug_convtx_balanced_focal_trainval_raw_visual_test_error_waveform_panels.png)

Bad-calibrated error waveform panel:

![badcal error waveform panel](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/but_capacity_error_gap/aug_convtx_balanced_focal_trainval_badcal_t005_visual_test_error_waveform_panels.png)

## Interpretation

The current failure is mostly a record/difficulty shift problem, not a simple "Transformer cannot learn SQI" problem. The Transformer learns `basSQI`, `qrs_band_ratio`, `baseline_step`, `flatline_ratio`, and detail-band targets reasonably well. It still under-learns `detector_agreement` and contact-loss style targets, and it does not yet convert the learned SQI targets into a stable three-way decision on the hard outlier record.

The next useful architecture change should not be another broad class-weight sweep. It should explicitly separate:

1. medium-vs-good under low-QRS / low-basSQI conditions,
2. bad-stress detection with strong non-bad specificity,
3. contact/baseline artifact recognition,
4. record-robust calibration rather than a single global bad threshold.

The most concrete next experiment is a BUT-only diagnostic Transformer with separate query heads for `QRS reliability`, `baseline/contact`, `detail/band`, and `bad-stress specificity`, plus a loss that penalizes medium->bad false positives when bad stress is activated.

## Artifacts

- Raw recovery report: `E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/but_direct_transformer_recovery/aug_convtx_balanced_focal_trainval_raw_report.md`
- Badcal recovery report: `E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/but_direct_transformer_recovery/aug_convtx_balanced_focal_trainval_badcal_t005_report.md`
- Split/capacity update: `E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/but_capacity_error_gap/but_split_seed_capacity_error_update_20260618.md`
