# PTB Bad Waveform Feature Match

Generated PTB bad waveforms are selected by recomputing the 39 waveform-computable SQI/morphology primitive features and matching them to BUT bad. This is stronger than sidecar-only 47-feature resampling because the actual waveform changes.

## Summary

- Target: `clean_margin_ge_5s_keep_outlier_bad_train_controlled` rows `4673`
- Base PTB train bad rows: `1319`
- Candidate generated rows: `25061`
- Matched rows: `1800` from `130` unique PTB sources
- Mean KS base -> BUT bad: `0.6801`
- Mean KS matched -> BUT bad: `0.7150`
- Top10 KS base -> BUT bad: `0.9893`
- Top10 KS matched -> BUT bad: `0.9935`
- Selected modes: `{'rightisland_osc': 1230, 'dropout': 304, 'detail': 176, 'smooth': 57, 'rough': 22, 'qrs_confuse': 5, 'reset': 5, 'mixed': 1}`
- Target region counts: `{'right_bad_island': 1529, 'outlier_low_confidence': 223, 'near_bad_boundary': 48}`

## Remaining Largest Matched KS

| feature | matched KS | target median | matched median |
| --- | ---: | ---: | ---: |
| hjorth_mobility | 1.0000 | 1.4336 | 1.1543 |
| higuchi_fd_proxy | 1.0000 | 1.0273 | 0.9893 |
| diff_zero_crossing_rate | 1.0000 | 0.7115 | 0.4143 |
| zero_crossing_rate | 1.0000 | 0.5741 | 0.4083 |
| band_15_30 | 0.9994 | 0.4573 | 0.9052 |
| sqi_bSQI | 0.9978 | 0.1099 | 0.5159 |
| fatal_or_score | 0.9903 | 1.1354 | 0.8467 |
| non_qrs_diff_p95 | 0.9891 | 2.7759 | 2.0234 |
| diff_abs_p95 | 0.9891 | 2.7759 | 2.0234 |
| wavelet_e4 | 0.9696 | 0.1041 | 0.0141 |
| sqi_fSQI | 0.9672 | 0.8892 | 0.9844 |
| wavelet_e3 | 0.9454 | 0.8349 | 0.9151 |

## Candidate Mode KS Summary

| mode | rows | mean KS | top10 KS | zcr KS | diff-zcr KS | band15-30 KS | non-qrs-diff KS | bSQI KS | flatline KS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dropout | 2160 | 0.6549 | 0.9615 | 1.0000 | 1.0000 | 0.9727 | 0.5527 | 0.6650 | 0.9181 |
| qrs_confuse | 2165 | 0.6773 | 0.9881 | 1.0000 | 1.0000 | 0.9364 | 0.9961 | 0.8728 | 0.7417 |
| reset | 2141 | 0.6828 | 0.9867 | 1.0000 | 1.0000 | 0.9695 | 0.9954 | 0.9268 | 0.7160 |
| detail | 2218 | 0.6879 | 0.9914 | 1.0000 | 1.0000 | 0.9662 | 0.9937 | 0.9910 | 0.6989 |
| mixed | 2134 | 0.6882 | 0.9875 | 1.0000 | 1.0000 | 0.8205 | 0.9650 | 0.8789 | 0.9803 |
| identity | 1319 | 0.6943 | 0.9914 | 1.0000 | 1.0000 | 0.9712 | 0.9912 | 0.9917 | 0.7183 |
| rough | 2144 | 0.6957 | 0.9926 | 1.0000 | 0.9995 | 0.9706 | 0.9968 | 0.9898 | 0.7087 |
| smooth | 2167 | 0.6966 | 0.9922 | 1.0000 | 0.9995 | 0.9746 | 0.9923 | 0.9949 | 0.7097 |
| record111 | 2083 | 0.6983 | 0.9832 | 1.0000 | 1.0000 | 0.7329 | 0.9509 | 0.8636 | 0.9766 |
| baseline | 2179 | 0.7195 | 0.9941 | 1.0000 | 0.9991 | 0.6782 | 0.9998 | 0.9923 | 0.9131 |
| flatstep_qrsvisible | 2166 | 0.7669 | 0.9985 | 1.0000 | 0.9995 | 0.9863 | 0.9788 | 0.6204 | 0.9984 |
| rightisland_osc | 2185 | 0.8517 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.6486 |

## Figures

![PCA](E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_waveform_feature_match\ptb_bad_waveform_feature_match_keepbad5s_train_controlled__80d4cb9c30bc_pca.png)

![Waveforms](E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_waveform_feature_match\ptb_bad_waveform_feature_match_keepbad5s_train_controlled__80d4cb9c30bc_waveforms.png)

## Files

- Matched signals: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_waveform_feature_match\ptb_bad_waveform_feature_match_keepbad5s_train_controlled__80d4cb9c30bc_signals.npz`
- Manifest: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_waveform_feature_match\ptb_bad_waveform_feature_match_keepbad5s_train_controlled__80d4cb9c30bc_manifest.csv`
- Feature KS: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_waveform_feature_match\ptb_bad_waveform_feature_match_keepbad5s_train_controlled__80d4cb9c30bc_feature_ks.csv`