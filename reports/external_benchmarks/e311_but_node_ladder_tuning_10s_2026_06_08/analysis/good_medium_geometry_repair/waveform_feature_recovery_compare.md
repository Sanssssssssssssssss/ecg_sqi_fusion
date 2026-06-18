# Waveform Feature Recovery Compare

Lower corr / higher z-MAE means the waveform student is not learning that teacher axis well.

## p20 worst corr: synthetic_test

| feature | corr | mae_z |
| --- | --- | --- |
| wavelet_e4 | -0.6517 | 0.7340 |
| hjorth_mobility | -0.4851 | 0.7814 |
| sqi_pSQI | -0.3359 | 0.6417 |
| wavelet_e3 | -0.3245 | 0.7125 |
| hjorth_activity | -0.3094 | 0.8502 |
| rms | -0.2669 | 0.7999 |
| mean_abs | 0.0300 | 0.9146 |
| pc2 | 0.0316 | 0.9959 |
| hjorth_complexity | 0.0748 | 0.6351 |
| fatal_or_score | 0.1170 | 0.9495 |
| qrs_visibility | 0.1614 | 0.8918 |
| sqi_basSQI | 0.1972 | 0.8974 |
| wavelet_e0 | 0.1977 | 0.8876 |
| pc4 | 0.2266 | 1.1652 |
| contact_loss_win_ratio | 0.2396 | 0.9179 |

## p20 worst corr: original

| feature | corr | mae_z |
| --- | --- | --- |
| wavelet_e4 | -0.8170 | 0.8854 |
| hjorth_mobility | -0.7795 | 0.8105 |
| hjorth_activity | -0.5391 | 0.9666 |
| wavelet_e3 | -0.4946 | 0.7107 |
| sqi_pSQI | -0.4843 | 0.6765 |
| rms | -0.3388 | 0.8786 |
| pc2 | -0.1313 | 0.6916 |
| hjorth_complexity | -0.0916 | 0.4636 |
| fatal_or_score | -0.0182 | 0.7220 |
| contact_loss_win_ratio | 0.1282 | 0.4012 |
| mean_abs | 0.1373 | 0.7509 |
| sqi_basSQI | 0.1669 | 0.5651 |
| pc4 | 0.1712 | 0.9834 |
| region_confidence | 0.1892 | 0.8407 |
| boundary_confidence | 0.1925 | 0.7768 |

## p20 worst MAE: synthetic_test

| feature | corr | mae_z |
| --- | --- | --- |
| pc4 | 0.2266 | 1.1652 |
| diff_abs_p95 | 0.3961 | 1.0421 |
| pc2 | 0.0316 | 0.9959 |
| sqi_kSQI | 0.4366 | 0.9629 |
| qrs_prom_p90 | 0.4402 | 0.9531 |
| fatal_or_score | 0.1170 | 0.9495 |
| contact_loss_win_ratio | 0.2396 | 0.9179 |
| mean_abs | 0.0300 | 0.9146 |
| sqi_basSQI | 0.1972 | 0.8974 |
| qrs_visibility | 0.1614 | 0.8918 |
| wavelet_e0 | 0.1977 | 0.8876 |
| hjorth_activity | -0.3094 | 0.8502 |
| baseline_step | 0.3207 | 0.8127 |
| qrs_band_ratio | 0.4420 | 0.8119 |
| sqi_sSQI | 0.4142 | 0.8089 |

## p20 worst MAE: original

| feature | corr | mae_z |
| --- | --- | --- |
| pc4 | 0.1712 | 0.9834 |
| hjorth_activity | -0.5391 | 0.9666 |
| qrs_visibility | 0.3966 | 0.8947 |
| wavelet_e4 | -0.8170 | 0.8854 |
| rms | -0.3388 | 0.8786 |
| region_confidence | 0.1892 | 0.8407 |
| hjorth_mobility | -0.7795 | 0.8105 |
| sqi_kSQI | 0.6214 | 0.7880 |
| low_amp_ratio | 0.5810 | 0.7849 |
| std | 0.5680 | 0.7805 |
| boundary_confidence | 0.1925 | 0.7768 |
| mean_abs | 0.1373 | 0.7509 |
| pc3 | 0.5323 | 0.7448 |
| wavelet_e0 | 0.2678 | 0.7290 |
| fatal_or_score | -0.0182 | 0.7220 |
