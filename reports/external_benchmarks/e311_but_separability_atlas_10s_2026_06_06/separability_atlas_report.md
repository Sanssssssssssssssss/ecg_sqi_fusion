# BUT 10s Three-Class Separability Atlas

## Executive read

- Medium independence interpretation: `medium_independent_or_mixed`.
- Medium off-axis ratio from the good-bad centroid line: `0.238`.
- Local medium neighbor purity k=30: `0.814`.
- Best interpretable probe macro-F1: `0.7471` via `linear_svc` / `protocol_train_to_test`.
- Best current generator/model context macro-F1: `0.7703` from `mg_qrs_confound_m_soft_midbad__bw_badstrong__cw1p00_1p55_1p70`.

## Top separating feature families

| family | n_features | mean_score | max_score | top_features |
| --- | --- | --- | --- | --- |
| Detail reliability | 9 | 0.7123 | 0.9616 | sample_entropy_proxy, higuchi_fd_proxy, non_qrs_diff_p95, diff_abs_median, non_qrs_rms_ratio, detail_instability |
| SQI | 7 | 0.5684 | 0.8028 | sqi_bSQI, sqi_pSQI, sqi_sSQI, sqi_kSQI, sqi_basSQI, sqi_fSQI |
| QRS detectability | 19 | 0.5104 | 0.7685 | template_corr, rr_count_detector_b, qrs_prom_p90, qrs_width_median, aggressive_peak_count, qrs_band_ratio |
| Motion/frequency | 18 | 0.6502 | 0.7444 | zero_crossing_rate, band_30_45, hjorth_mobility, wavelet_e4, diff_zero_crossing_rate, band_5_15 |
| Contact/flat/fatal | 7 | 0.3864 | 0.6835 | flatline_ratio, baseline_step, low_amp_ratio, local_rms_cv, fatal_or_score, contact_loss_win_ratio |
| Amplitude/global | 4 | 0.5765 | 0.6638 | rms, std, ptp_p99_p01, mean_abs |

## Top individual dimensions

| feature | family | separability_score | max_ks | max_abs_cliffs | mutual_info | mean_probe_importance |
| --- | --- | --- | --- | --- | --- | --- |
| sample_entropy_proxy | Detail reliability | 0.9616 | 0.9432 | 0.9645 | 0.6617 | 0.5773 |
| higuchi_fd_proxy | Detail reliability | 0.8406 | 0.9364 | 0.9241 | 0.6154 | 0.2951 |
| sqi_bSQI | SQI | 0.8028 | 0.9568 | 0.9937 | 0.5075 | 0.1379 |
| non_qrs_diff_p95 | Detail reliability | 0.778 | 0.9337 | 0.9414 | 0.5184 | 0.1673 |
| template_corr | QRS detectability | 0.7685 | 0.967 | 0.989 | 0.4886 | 0.2202 |
| diff_abs_median | Detail reliability | 0.7625 | 0.934 | 0.9072 | 0.5354 | 0.1283 |
| zero_crossing_rate | Motion/frequency | 0.7444 | 0.9337 | 0.912 | 0.485 | 0.1127 |
| band_30_45 | Motion/frequency | 0.7255 | 0.9336 | 0.9103 | 0.4698 | 0.1791 |
| hjorth_mobility | Motion/frequency | 0.7143 | 0.9336 | 0.8924 | 0.4723 | 0.1387 |
| sqi_pSQI | SQI | 0.7115 | 0.9325 | 0.8891 | 0.4554 | 0.1129 |
| wavelet_e4 | Motion/frequency | 0.7047 | 0.9336 | 0.8986 | 0.4519 | 0.1052 |
| diff_zero_crossing_rate | Motion/frequency | 0.687 | 0.9316 | 0.9028 | 0.5033 | 0.06406 |
| flatline_ratio | Contact/flat/fatal | 0.6835 | 0.9376 | 0.9189 | 0.5885 | 0.04344 |
| band_5_15 | Motion/frequency | 0.6828 | 0.9127 | 0.9804 | 0.3938 | 0.09071 |
| non_qrs_rms_ratio | Detail reliability | 0.6823 | 0.9231 | 0.9711 | 0.4374 | 0.06361 |
| spectral_entropy | Motion/frequency | 0.6823 | 0.8918 | 0.9214 | 0.39 | 0.1321 |
| band_15_30 | Motion/frequency | 0.681 | 0.9086 | 0.8773 | 0.4483 | 0.08235 |
| sqi_sSQI | SQI | 0.6709 | 0.9467 | 0.9583 | 0.4445 | 0.0512 |
| rr_count_detector_b | QRS detectability | 0.6644 | 0.9172 | 0.9359 | 0.4133 | 0.09593 |
| rms | Amplitude/global | 0.6638 | 0.9153 | 0.977 | 0.3896 | 0.1072 |

## Boundary-specific evidence

### Good vs medium

| feature | median_delta_a_minus_b | cliffs_delta | ks | wasserstein | mutual_info |
| --- | --- | --- | --- | --- | --- |
| sample_entropy_proxy | -0.1599 | -0.8361 | 0.7103 | 0.1606 | 0.6617 |
| aggressive_peak_count | -29 | -0.703 | 0.6197 | 22.51 | 0.4598 |
| higuchi_fd_proxy | -0.1049 | -0.6423 | 0.5996 | 0.1154 | 0.6154 |
| spurious_peak_density | -2.1 | -0.6763 | 0.5988 | 1.818 | 0.3644 |
| flatline_ratio | 0.1938 | 0.6987 | 0.5918 | 0.1534 | 0.5885 |
| non_qrs_diff_p95 | -0.04007 | -0.5276 | 0.489 | 0.04504 | 0.5184 |
| diff_abs_median | -0.01173 | -0.5614 | 0.489 | 0.01101 | 0.5354 |
| rr_count_detector_b | -5 | -0.6155 | 0.4865 | 4.035 | 0.4133 |
| diff_zero_crossing_rate | -0.08654 | -0.474 | 0.4476 | 0.06768 | 0.5033 |
| sqi_sSQI | 1.361 | 0.5323 | 0.4365 | 1.453 | 0.4445 |
| low_amp_ratio | 0.1212 | 0.5353 | 0.4356 | 0.1092 | 0.3641 |
| qrs_visibility | 0.2478 | 0.527 | 0.427 | 0.2285 | 0.401 |

### Medium vs bad

| feature | median_delta_a_minus_b | cliffs_delta | ks | wasserstein | mutual_info |
| --- | --- | --- | --- | --- | --- |
| hjorth_complexity | 0.5999 | 0.8902 | 0.9334 | 1.496 | 0.4324 |
| wavelet_e4 | -0.2368 | -0.8986 | 0.9334 | 0.228 | 0.4519 |
| hjorth_mobility | -0.8117 | -0.8924 | 0.9333 | 0.8276 | 0.4723 |
| higuchi_fd_proxy | -0.5234 | -0.8971 | 0.9332 | 0.4848 | 0.6154 |
| zero_crossing_rate | -0.4692 | -0.8963 | 0.9329 | 0.4366 | 0.485 |
| diff_abs_median | -0.1525 | -0.89 | 0.9321 | 0.1431 | 0.5354 |
| sample_entropy_proxy | -0.4106 | -0.9191 | 0.932 | 0.3884 | 0.6617 |
| non_qrs_diff_p95 | -0.2835 | -0.9122 | 0.9317 | 0.2611 | 0.5184 |
| detail_instability | 0.8839 | 0.9034 | 0.9313 | 0.8577 | 0.4863 |
| sqi_pSQI | 0.5449 | 0.8862 | 0.9312 | 0.5049 | 0.4554 |
| flatline_ratio | 0.09528 | 0.8927 | 0.931 | 0.1125 | 0.5885 |
| band_30_45 | -0.0787 | -0.8991 | 0.924 | 0.07655 | 0.4698 |

### Good vs bad

| feature | median_delta_a_minus_b | cliffs_delta | ks | wasserstein | mutual_info |
| --- | --- | --- | --- | --- | --- |
| template_corr | 0.478 | 0.989 | 0.967 | 0.4846 | 0.4886 |
| sqi_bSQI | 0.9 | 0.9937 | 0.9568 | 0.8227 | 0.5075 |
| sqi_sSQI | 4.055 | 0.9583 | 0.9467 | 3.75 | 0.4445 |
| sample_entropy_proxy | -0.5704 | -0.9645 | 0.9432 | 0.5489 | 0.6617 |
| flatline_ratio | 0.289 | 0.9189 | 0.9376 | 0.2643 | 0.5885 |
| higuchi_fd_proxy | -0.6284 | -0.9241 | 0.9364 | 0.5959 | 0.6154 |
| diff_abs_median | -0.1642 | -0.9072 | 0.934 | 0.1541 | 0.5354 |
| zero_crossing_rate | -0.5092 | -0.912 | 0.9337 | 0.4737 | 0.485 |
| non_qrs_diff_p95 | -0.3235 | -0.9414 | 0.9337 | 0.3061 | 0.5184 |
| band_30_45 | -0.08589 | -0.9103 | 0.9336 | 0.08466 | 0.4698 |
| hjorth_mobility | -0.8366 | -0.8827 | 0.9336 | 0.7981 | 0.4723 |
| wavelet_e4 | -0.2383 | -0.8977 | 0.9336 | 0.2294 | 0.4519 |

## Current answer

- The classes are not best described by a single SNR line. BUT medium is at least mixed/partly independent: QRS-usable windows can be medium even when some global noise features overlap good or bad.
- The most useful next generator target is a small set of axes: QRS detectability, non-QRS/detail reliability, fatal contact/flat events, baseline/HF motion, and SQI consistency.
- SQI should remain a diagnostic branch because prior SQI gap analysis showed strong BUT-vs-PTB domain signature and class-direction flips.

## Files

- Output root: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_separability_atlas_10s_2026_06_06`
- Report root: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_separability_atlas_10s_2026_06_06`