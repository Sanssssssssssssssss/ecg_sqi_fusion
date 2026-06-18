# N17043 Bad Split And Ambiguous Boundary Diagnostic

Selection remains Clean/SemiClean/node diagnostic only. Original BUT remains report-only.

## Main Finding

The apparent test-split bad recall collapse is a domain/split issue, not evidence that bad is globally unlearnable.

- Train bad: record 105001, `right_bad_island`, 3964 rows.
- Test bad: record 122001, `near_bad_boundary` / `bad_outlier`, 119 rows.
- The current best N17043 ordinary checkpoint predicts those 119 test-bad rows as medium, while the all-node bad recall stays high because train bad dominates the bad support.

## High-Confidence Boundary Option

- `boundary_confidence >= 0.6` and `pca_margin >= 1.2` keeps 19,600 of 31,755 rows and passes all-node gate: acc 0.9509, good/medium/bad recall 0.9467/0.9440/0.9706.
- Excluded rows should be called an ambiguous boundary/stress bucket, not silently removed from the main target.

## Strongest Test-Bad vs Train-Bad Feature Gaps

| feature | test_bad_median | train_bad_median | median_delta_left_minus_right | test_bad_p10 | test_bad_p90 | train_bad_p10 | train_bad_p90 | ks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qrs_band_ratio | 0.331602 | 0.811202 | -0.4796 | 0.303541 | 0.36873 | 0.794037 | 0.82629 | 1 |
| detail_instability | 0.28289 | 0.2262 | 0.05669 | 0.270833 | 0.295698 | 0.218893 | 0.233998 | 1 |
| baseline_step | 0.24753 | 0.0270451 | 0.220485 | 0.141538 | 0.35973 | 0.0164437 | 0.0406785 | 1 |
| qrs_visibility | 0.104145 | 0.247852 | -0.143707 | 0.0939146 | 0.118055 | 0.235192 | 0.262239 | 1 |
| boundary_confidence | 0.39068 | 0.757167 | -0.366487 | 0.386462 | 0.394615 | 0.561653 | 1.11285 | 1 |
| pca_margin | 5.34877 | 10.954 | -5.60522 | 4.89902 | 5.80661 | 10.5857 | 11.2153 | 1 |
| band_15_30 | 0.321031 | 0.834417 | -0.513386 | 0.292955 | 0.357589 | 0.82002 | 0.848255 | 1 |
| band_30_45 | 0.276748 | 0.103568 | 0.17318 | 0.242382 | 0.33147 | 0.0922529 | 0.115988 | 1 |
| pc4 | -1.55688 | 0.629651 | -2.18653 | -1.94362 | -1.19736 | 0.293801 | 0.916399 | 1 |
| sqi_bSQI | 0.482759 | 0 | 0.482759 | 0.387097 | 0.578205 | 0 | 0.0416667 | 0.998991 |

## Test-Bad vs Test-Medium Gaps

| feature | test_bad_median | test_medium_median | median_delta_left_minus_right | test_bad_p10 | test_bad_p90 | test_medium_p10 | test_medium_p90 | ks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| band_30_45 | 0.276748 | 0.0278649 | 0.248883 | 0.242382 | 0.33147 | 0.00204704 | 0.0494571 | 1 |
| pca_margin | 5.34877 | 1.59465 | 3.75413 | 4.89902 | 5.80661 | -0.162147 | 3.08247 | 1 |
| pc1 | 9.00538 | -1.10392 | 10.1093 | 8.5663 | 9.44357 | -4.90996 | 0.40731 | 1 |
| flatline_ratio | 0.00880705 | 0.132906 | -0.124099 | 0.006245 | 0.0120096 | 0.06245 | 0.426741 | 0.999774 |
| non_qrs_diff_p95 | 0.417235 | 0.0717831 | 0.345452 | 0.390019 | 0.444514 | 0.0144163 | 0.151673 | 0.999322 |
| detail_instability | 0.28289 | 0.792029 | -0.509139 | 0.270833 | 0.295698 | 0.479274 | 1.55724 | 0.998644 |
| template_corr | 0.196245 | 0.475716 | -0.27947 | 0.159722 | 0.238426 | 0.338893 | 0.932377 | 0.938414 |
| non_qrs_rms_ratio | 0.921674 | 0.561775 | 0.359899 | 0.88668 | 0.978802 | 0.296091 | 0.858594 | 0.893992 |
| diff_abs_p95 | 0.437352 | 0.111666 | 0.325686 | 0.419903 | 0.453328 | 0.022 | 0.424706 | 0.873883 |
| band_15_30 | 0.321031 | 0.162438 | 0.158594 | 0.292955 | 0.357589 | 0.0140389 | 0.281696 | 0.86978 |

## Figures

![PCA](E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\n17043_bad_split_domain_pca.png)

![Feature boxplots](E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\n17043_bad_split_feature_boxplots.png)

![Raw waveforms](E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\n17043_bad_split_raw_waveforms.png)

## Recommended Next Experiment

Try one compact near-boundary bad-domain block that mimics 122001-like low-amplitude/near-boundary bad, while preserving the 19.6k high-confidence good/medium target as the primary learnable body.