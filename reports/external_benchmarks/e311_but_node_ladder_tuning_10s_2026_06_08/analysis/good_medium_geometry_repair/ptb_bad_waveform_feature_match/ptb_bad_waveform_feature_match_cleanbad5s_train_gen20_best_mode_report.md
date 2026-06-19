# PTB Bad Waveform Feature Match

Generated PTB bad waveforms are selected by recomputing the 39 waveform-computable SQI/morphology primitive features and matching them to BUT bad. This is stronger than sidecar-only 47-feature resampling because the actual waveform changes.

## Summary

- Target: `clean_margin_ge_5s_drop_outlier_bad_train` rows `3963`
- Base PTB train bad rows: `1319`
- Candidate generated rows: `27699`
- Matched rows: `1450` from `882` unique PTB sources
- Mean KS base -> BUT bad: `0.6897`
- Mean KS matched -> BUT bad: `0.6617`
- Top10 KS base -> BUT bad: `0.9922`
- Top10 KS matched -> BUT bad: `0.9662`
- Selected modes: `{'dropout': 1450}`
- Target region counts: `{'right_bad_island': 1450}`

## Remaining Largest Matched KS

| feature | matched KS | target median | matched median |
| --- | ---: | ---: | ---: |
| hjorth_mobility | 1.0000 | 1.4339 | 0.9751 |
| diff_zero_crossing_rate | 1.0000 | 0.7123 | 0.4351 |
| non_qrs_rms_ratio | 1.0000 | 1.0345 | 0.9805 |
| zero_crossing_rate | 1.0000 | 0.5749 | 0.3155 |
| wavelet_e0 | 0.9785 | 0.0002 | 0.0051 |
| band_15_30 | 0.9752 | 0.4578 | 0.8553 |
| higuchi_fd_proxy | 0.9557 | 1.0274 | 0.9909 |
| detector_agreement | 0.9209 | 0.0000 | 0.5000 |
| sqi_iSQI | 0.9209 | 0.0000 | 0.5000 |
| hjorth_complexity | 0.9111 | 1.2145 | 1.1106 |
| flatline_ratio | 0.8801 | 0.0056 | 0.0200 |
| wavelet_e2 | 0.8747 | 0.0499 | 0.0204 |

## Candidate Mode KS Summary

| mode | rows | mean KS | top10 KS | zcr KS | diff-zcr KS | band15-30 KS | non-qrs-diff KS | bSQI KS | flatline KS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dropout | 1832 | 0.6613 | 0.9659 | 1.0000 | 1.0000 | 0.9733 | 0.5687 | 0.7494 | 0.8789 |
| reset | 1698 | 0.6788 | 0.9890 | 1.0000 | 1.0000 | 0.9368 | 0.9902 | 0.8197 | 0.7248 |
| qrs_confuse | 1757 | 0.6958 | 0.9988 | 1.0000 | 1.0000 | 0.7558 | 0.9980 | 0.6819 | 0.7430 |
| highzcr | 1814 | 0.6997 | 0.9930 | 1.0000 | 1.0000 | 0.9604 | 0.9906 | 0.9934 | 0.7225 |
| detail | 1692 | 0.6998 | 0.9944 | 1.0000 | 1.0000 | 0.9687 | 0.9919 | 0.9970 | 0.7414 |
| identity | 1319 | 0.7034 | 0.9943 | 1.0000 | 1.0000 | 0.9712 | 0.9914 | 0.9917 | 0.7243 |
| smooth | 1827 | 0.7056 | 0.9948 | 1.0000 | 1.0000 | 0.9715 | 0.9922 | 0.9957 | 0.7074 |
| rough | 1782 | 0.7056 | 0.9979 | 1.0000 | 1.0000 | 0.9583 | 0.9981 | 0.9933 | 0.7398 |
| cleanhf | 1740 | 0.7213 | 0.9924 | 0.5425 | 0.8229 | 0.9633 | 0.7692 | 1.0000 | 0.2144 |
| cleanhf_qrsdrop | 1725 | 0.7241 | 0.9937 | 0.5941 | 0.8148 | 0.9775 | 0.8081 | 1.0000 | 0.2266 |
| mixed | 1739 | 0.7266 | 0.9985 | 1.0000 | 1.0000 | 0.9548 | 0.9896 | 0.8795 | 0.9893 |
| record111 | 1767 | 0.7270 | 0.9978 | 1.0000 | 1.0000 | 0.8967 | 0.9848 | 0.8920 | 0.9850 |

## Figures

![PCA](E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_waveform_feature_match\ptb_bad_waveform_feature_match_cleanbad5s_train_gen20_best_mode_pca.png)

![Waveforms](E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_waveform_feature_match\ptb_bad_waveform_feature_match_cleanbad5s_train_gen20_best_mode_waveforms.png)

## Files

- Matched signals: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_waveform_feature_match\ptb_bad_waveform_feature_match_cleanbad5s_train_gen20_best_mode_signals.npz`
- Manifest: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_waveform_feature_match\ptb_bad_waveform_feature_match_cleanbad5s_train_gen20_best_mode_manifest.csv`
- Feature KS: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_waveform_feature_match\ptb_bad_waveform_feature_match_cleanbad5s_train_gen20_best_mode_feature_ks.csv`