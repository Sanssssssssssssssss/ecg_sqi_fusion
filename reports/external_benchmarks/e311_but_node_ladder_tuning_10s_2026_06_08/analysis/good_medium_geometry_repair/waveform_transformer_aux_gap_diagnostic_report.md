# Waveform Transformer Aux Gap Diagnostic

Question: does the waveform Transformer learn the SQI/geometry quantities that make the tabular MLP strong?

## Split Metrics

| split | acc | macro-F1 | good R | medium R | bad R |
|---|---:|---:|---:|---:|---:|
| train | 0.924020 | 0.929184 | 0.894081 | 0.928817 | 0.995617 |
| val | 0.980121 | 0.960094 | 0.981424 | 0.961905 | 0.987952 |
| test | 0.811018 | 0.730436 | 0.960714 | 0.725938 | 0.401460 |

## Highest-Impact Missing Auxiliary Features On Test

| feature | ranking score | MAE z | corr | true GM AUC | pred GM AUC | true bad AUC | pred bad AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| pca_margin | 2.872 | 0.225 | 0.610 | 0.509 | 0.520 | 0.710 | 0.523 |
| sqi_iSQI | 1.000 | 0.022 | nan | 0.500 | 0.774 | 0.500 | 0.580 |
| knn_label_purity | 1.707 | 1.075 | 0.488 | 0.837 | 0.809 | 0.804 | 0.536 |
| detector_agreement | 1.518 | 0.468 | 0.535 | 0.530 | 0.768 | 0.695 | 0.621 |
| pc3 | 2.022 | 0.594 | 0.651 | 0.780 | 0.780 | 0.610 | 0.537 |
| qrs_visibility | 2.060 | 0.351 | 0.661 | 0.738 | 0.698 | 0.698 | 0.745 |
| sqi_bSQI | 1.959 | 0.306 | 0.691 | 0.544 | 0.641 | 0.898 | 0.912 |
| sqi_fSQI | 2.117 | 1.020 | 0.738 | 0.755 | 0.722 | 0.589 | 0.580 |
| contact_loss_win_ratio | 1.025 | 1.624 | 0.467 | 0.592 | 0.771 | 0.542 | 0.506 |
| sqi_pSQI | 1.912 | 0.221 | 0.718 | 0.626 | 0.618 | 0.562 | 0.567 |
| fatal_or_score | 1.443 | 1.263 | 0.627 | 0.755 | 0.939 | 0.538 | 0.796 |
| template_corr | 1.836 | 0.480 | 0.720 | 0.690 | 0.753 | 0.945 | 0.960 |
| mean_abs | 1.755 | 0.622 | 0.719 | 0.671 | 0.715 | 0.590 | 0.679 |
| diff_zero_crossing_rate | 2.209 | 0.310 | 0.813 | 0.700 | 0.724 | 0.585 | 0.613 |
| baseline_step | 2.063 | 0.702 | 0.809 | 0.611 | 0.737 | 0.528 | 0.525 |
| boundary_confidence | 1.410 | 0.582 | 0.771 | 0.549 | 0.516 | 0.929 | 0.584 |

## Interpretation

- If a feature has high tabular ranking but low aux correlation/high MAE, the waveform encoder is not encoding the quantity the MLP uses.
- If true AUC is strong but predicted-feature AUC is weak, the feature is learnable/useful in tabular form but not recovered by the current Transformer representation.
- This separates architecture/representation failure from simple class-weight or threshold failure.

Feature gap CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_transformer_aux_feature_gap.csv`
Error feature CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_transformer_aux_error_feature_gap.csv`