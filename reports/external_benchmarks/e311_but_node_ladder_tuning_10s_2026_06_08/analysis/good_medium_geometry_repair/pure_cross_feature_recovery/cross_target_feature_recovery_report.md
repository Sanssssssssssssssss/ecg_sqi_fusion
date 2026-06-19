# Pure Cross Target Feature Recovery

Created: 2026-06-19T17:02:58

Auxiliary feature recovery on the held-out opposite domain. Features are training-time targets only, not inference inputs.

## ptb_train_to_clean_but_test

| feature | corr_norm | mae_norm |
| --- | --- | --- |
| qrs_band_ratio | -0.4762 | 0.5523 |
| sqi_basSQI | -0.05735 | 0.4838 |
| qrs_visibility | 0.07052 | 0.9729 |
| baseline_step | 0.2328 | 0.5259 |
| template_corr | 0.2679 | 0.7259 |
| detector_agreement | 0.4117 | 0.6351 |
| sqi_bSQI | 0.6515 | 0.2094 |
| sqi_pSQI | 0.7438 | 0.207 |
| amplitude_entropy | 0.8133 | 0.4824 |
| flatline_ratio | 0.8262 | 0.5497 |
| non_qrs_diff_p95 | 0.829 | 0.3445 |
| sqi_kSQI | 0.8339 | 1.111 |
| band_30_45 | 0.8802 | 0.7478 |

Lowest-correlation features:

| feature | corr_norm | mae_norm | is_key |
| --- | --- | --- | --- |
| qrs_band_ratio | -0.4762 | 0.5523 | True |
| mean_abs | -0.2784 | 0.7883 | False |
| sqi_basSQI | -0.05735 | 0.4838 | True |
| wavelet_e2 | -0.0465 | 0.5131 | False |
| qrs_visibility | 0.07052 | 0.9729 | True |
| hjorth_complexity | 0.1377 | 0.1857 | False |
| wavelet_e0 | 0.1541 | 0.5235 | False |
| band_15_30 | 0.1992 | 0.2552 | False |
| wavelet_e1 | 0.2256 | 0.5022 | False |
| baseline_step | 0.2328 | 0.5259 | True |
| template_corr | 0.2679 | 0.7259 | True |
| detector_agreement | 0.4117 | 0.6351 | True |

## clean_but_train_to_ptb_test

| feature | corr_norm | mae_norm |
| --- | --- | --- |
| sqi_basSQI | 0.09673 | 2.494 |
| qrs_visibility | 0.1442 | 0.7563 |
| detector_agreement | 0.2023 | 0.6677 |
| baseline_step | 0.2033 | 1.427 |
| sqi_kSQI | 0.3379 | 0.8867 |
| qrs_band_ratio | 0.3444 | 1.169 |
| amplitude_entropy | 0.3536 | 0.7774 |
| template_corr | 0.3981 | 0.642 |
| flatline_ratio | 0.4488 | 0.7361 |
| sqi_bSQI | 0.604 | 0.4228 |
| band_30_45 | 0.6525 | 0.4291 |
| non_qrs_diff_p95 | 0.6705 | 0.4563 |
| sqi_pSQI | 0.7282 | 0.3558 |

Lowest-correlation features:

| feature | corr_norm | mae_norm | is_key |
| --- | --- | --- | --- |
| contact_loss_win_ratio | -0.09958 | 0.03711 | False |
| mean_abs | 0.03929 | 1.038 | False |
| sqi_basSQI | 0.09673 | 2.494 | True |
| hjorth_activity | 0.1111 | 0.9762 | False |
| qrs_visibility | 0.1442 | 0.7563 | True |
| std | 0.1499 | 0.9527 | False |
| rms | 0.1626 | 0.9221 | False |
| fatal_or_score | 0.1772 | 1.195 | False |
| ptp_p99_p01 | 0.1801 | 0.9446 | False |
| detector_agreement | 0.2023 | 0.6677 | True |
| hjorth_complexity | 0.2024 | 3.457 | False |
| baseline_step | 0.2033 | 1.427 | True |
