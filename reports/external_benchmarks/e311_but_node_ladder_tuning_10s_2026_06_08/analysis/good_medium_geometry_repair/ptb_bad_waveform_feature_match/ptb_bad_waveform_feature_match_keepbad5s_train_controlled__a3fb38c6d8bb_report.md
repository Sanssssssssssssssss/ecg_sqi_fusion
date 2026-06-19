# PTB Bad Waveform Feature Match

Generated PTB bad waveforms are selected by recomputing the 39 waveform-computable SQI/morphology primitive features and matching them to BUT bad. This is stronger than sidecar-only 47-feature resampling because the actual waveform changes.

## Summary

- Target: `clean_margin_ge_5s_keep_outlier_bad_train_controlled` rows `4780`
- Base PTB train bad rows: `9229`
- Candidate generated rows: `55374`
- Matched rows: `2200` from `1950` unique PTB sources
- Mean KS base -> BUT bad: `0.8022`
- Mean KS matched -> BUT bad: `0.7042`
- Top10 KS base -> BUT bad: `0.9500`
- Top10 KS matched -> BUT bad: `0.9481`
- Selected modes: `{'baseline': 259, 'mixed': 241, 'identity': 226, 'reset': 220, 'dropout': 214, 'detail': 202, 'smooth': 178, 'rough': 170, 'flatstep_qrsvisible': 134, 'qrs_confuse': 128, 'record111': 121, 'rightisland_osc': 107}`
- Target region counts: `{'right_bad_island': 1836, 'outlier_low_confidence': 306, 'near_bad_boundary': 58}`

## Remaining Largest Matched KS

| feature | matched KS | target median | matched median |
| --- | ---: | ---: | ---: |
| zero_crossing_rate | 1.0000 | 0.5741 | 0.1461 |
| diff_zero_crossing_rate | 1.0000 | 0.7115 | 0.4343 |
| hjorth_mobility | 1.0000 | 1.4336 | 0.5961 |
| higuchi_fd_proxy | 0.9889 | 1.0273 | 0.9407 |
| non_qrs_rms_ratio | 0.9678 | 1.0344 | 0.7844 |
| sqi_bSQI | 0.9432 | 0.1096 | 0.8199 |
| fatal_or_score | 0.9164 | 1.1354 | 0.6445 |
| diff_abs_p95 | 0.8900 | 2.7757 | 1.6967 |
| non_qrs_diff_p95 | 0.8900 | 2.7757 | 1.6967 |
| detector_agreement | 0.8842 | 0.0000 | 0.5833 |
| sqi_iSQI | 0.8842 | 0.0000 | 0.5833 |
| wavelet_e0 | 0.8679 | 0.0002 | 0.0339 |

## Candidate Mode KS Summary

| mode | rows | mean KS | top10 KS | zcr KS | diff-zcr KS | band15-30 KS | non-qrs-diff KS | bSQI KS | flatline KS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 4078 | 0.7755 | 0.9899 | 0.9998 | 0.9993 | 0.9276 | 0.9719 | 0.9875 | 0.9789 |
| flatstep_qrsvisible | 4139 | 0.7829 | 0.9994 | 1.0000 | 0.9992 | 0.9977 | 0.9943 | 0.9453 | 0.9994 |
| detail | 4199 | 0.7952 | 0.9533 | 1.0000 | 1.0000 | 0.8265 | 0.7762 | 0.8790 | 0.9116 |
| identity | 9229 | 0.8006 | 0.9489 | 0.9999 | 0.9998 | 0.8349 | 0.7570 | 0.8399 | 0.9213 |
| rough | 4297 | 0.8017 | 0.9601 | 1.0000 | 1.0000 | 0.8425 | 0.7871 | 0.8832 | 0.9170 |
| mixed | 4212 | 0.8035 | 0.9951 | 1.0000 | 1.0000 | 0.9496 | 0.9858 | 0.9877 | 0.9937 |
| smooth | 4192 | 0.8037 | 0.9563 | 1.0000 | 0.9989 | 0.8415 | 0.8605 | 0.9113 | 0.9181 |
| rightisland_osc | 4172 | 0.8052 | 1.0000 | 1.0000 | 1.0000 | 0.9966 | 1.0000 | 1.0000 | 0.6535 |
| reset | 4218 | 0.8090 | 0.9611 | 1.0000 | 0.9998 | 0.8359 | 0.7565 | 0.8464 | 0.9183 |
| record111 | 4256 | 0.8149 | 0.9916 | 1.0000 | 1.0000 | 0.9366 | 0.9709 | 0.9764 | 0.9956 |
| qrs_confuse | 4203 | 0.8185 | 0.9687 | 1.0000 | 1.0000 | 0.8459 | 0.7949 | 0.8578 | 0.9262 |
| dropout | 4179 | 0.8206 | 0.9695 | 1.0000 | 0.9978 | 0.8373 | 0.7206 | 0.7652 | 0.9706 |

## Figures

![PCA](E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_waveform_feature_match\ptb_bad_waveform_feature_match_keepbad5s_train_controlled__a3fb38c6d8bb_pca.png)

![Waveforms](E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_waveform_feature_match\ptb_bad_waveform_feature_match_keepbad5s_train_controlled__a3fb38c6d8bb_waveforms.png)

## Files

- Matched signals: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_waveform_feature_match\ptb_bad_waveform_feature_match_keepbad5s_train_controlled__a3fb38c6d8bb_signals.npz`
- Manifest: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_waveform_feature_match\ptb_bad_waveform_feature_match_keepbad5s_train_controlled__a3fb38c6d8bb_manifest.csv`
- Feature KS: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_waveform_feature_match\ptb_bad_waveform_feature_match_keepbad5s_train_controlled__a3fb38c6d8bb_feature_ks.csv`