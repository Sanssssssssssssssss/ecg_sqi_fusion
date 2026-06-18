# Simple Feature Bootstrap Stability

The shallow feature rule was re-fit on 120 bootstrap samples of the N7188 Clean/SemiClean node train+val set. Original BUT is report-only.

## Metric Stability
| dataset | acc_median | acc_p10 | acc_p90 | good_recall_median | medium_recall_median | bad_recall_median |
| --- | --- | --- | --- | --- | --- | --- |
| node_test | 0.998506 | 0.968850 | 0.999751 | 0.997859 | 1.000000 | 1.000000 |
| original_test | 0.799988 | 0.769730 | 0.853156 | 0.685577 | 0.968821 | 0.289538 |
| original_all | 0.830592 | 0.822846 | 0.839990 | 0.722467 | 0.962128 | 0.932261 |

## Feature Usage
| feature | trees_used | usage_rate | split_count |
| --- | --- | --- | --- |
| pc1 | 120 | 1.000000 | 314 |
| qrs_band_ratio | 70 | 0.583333 | 85 |
| qrs_visibility | 58 | 0.483333 | 60 |
| qrs_prom_p90 | 51 | 0.425000 | 51 |
| band_15_30 | 30 | 0.250000 | 35 |
| amplitude_entropy | 18 | 0.150000 | 18 |
| sqi_sSQI | 13 | 0.108333 | 13 |
| sqi_basSQI | 11 | 0.091667 | 13 |
| low_amp_ratio | 11 | 0.091667 | 11 |
| flatline_ratio | 11 | 0.091667 | 11 |
| pc2 | 10 | 0.083333 | 12 |
| baseline_step | 8 | 0.066667 | 8 |

## Threshold Stability
| feature | usage_count | median | p10 | p90 |
| --- | --- | --- | --- | --- |
| pc1 | 314 | -2.113896 | -2.415049 | 6.237048 |
| qrs_band_ratio | 85 | 0.354245 | 0.343286 | 0.747672 |
| qrs_visibility | 60 | 0.518383 | 0.208190 | 0.523042 |
| qrs_prom_p90 | 51 | 5.029391 | 4.849511 | 5.184185 |
| band_15_30 | 35 | 0.366083 | 0.122739 | 0.803696 |
| amplitude_entropy | 18 | 0.580760 | 0.529799 | 0.586027 |
| sqi_basSQI | 13 | 0.998920 | 0.810794 | 0.999269 |
| sqi_sSQI | 13 | 2.006327 | -0.302726 | 2.183212 |
| pc2 | 12 | 2.463966 | -4.328381 | 5.514682 |
| flatline_ratio | 11 | 0.002002 | 0.002002 | 0.211769 |

## Interpretation
- Stable use of `pc1` means the major good/medium/bad geometry is low-dimensional, not a fragile neural artifact.
- Stable use of `qrs_prom_p90` means the good/medium overlap needs a morphology cue, not only SNR-like noise strength.
- Original performance remains lower than clean-node performance, so the remaining problem is domain/label transfer plus bad outlier stress, not missing clean-node separability.

![Bootstrap thresholds](E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\simple_feature_bootstrap_thresholds.png)
