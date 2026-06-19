# PTB Bad Waveform Feature Match

Generated PTB bad waveforms are selected by recomputing the 39 waveform-computable SQI/morphology primitive features and matching them to BUT bad. This is stronger than sidecar-only 47-feature resampling because the actual waveform changes.

## Summary

- Target: `but_bad_test` rows `411`
- Base PTB train bad rows: `1319`
- Candidate generated rows: `17147`
- Matched rows: `1433` from `897` unique PTB sources
- Mean KS base -> BUT bad: `0.5975`
- Mean KS matched -> BUT bad: `0.3210`
- Top10 KS base -> BUT bad: `0.8889`
- Top10 KS matched -> BUT bad: `0.5166`
- Selected modes: `{'mixed': 1433}`
- Target region counts: `{'outlier_low_confidence': 996, 'near_bad_boundary': 437}`

## Remaining Largest Matched KS

| feature | matched KS | target median | matched median |
| --- | ---: | ---: | ---: |
| flatline_ratio | 0.6398 | 0.2458 | 0.0424 |
| sample_entropy_proxy | 0.5654 | 0.6240 | 0.3821 |
| sqi_sSQI | 0.5654 | 0.3760 | 0.6179 |
| wavelet_e2 | 0.5564 | 0.1854 | 0.0640 |
| sqi_fSQI | 0.5508 | 0.7576 | 0.8640 |
| diff_zero_crossing_rate | 0.5137 | 0.3510 | 0.4231 |
| zero_crossing_rate | 0.4575 | 0.0440 | 0.1329 |
| sqi_basSQI | 0.4436 | 0.7747 | 0.8379 |
| baseline_step | 0.4436 | 0.0727 | 0.0484 |
| wavelet_e1 | 0.4299 | 0.2817 | 0.1804 |
| wavelet_e3 | 0.4237 | 0.0582 | 0.2134 |
| sqi_bSQI | 0.3835 | 0.7021 | 0.4696 |

## Candidate Mode KS Summary

| mode | rows | mean KS | top10 KS | zcr KS | diff-zcr KS | band15-30 KS | non-qrs-diff KS | bSQI KS | flatline KS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mixed | 1433 | 0.3210 | 0.5166 | 0.4575 | 0.5137 | 0.2258 | 0.2818 | 0.3835 | 0.6398 |
| stress | 1432 | 0.3214 | 0.5158 | 0.5484 | 0.5144 | 0.3422 | 0.2267 | 0.4482 | 0.6666 |
| record111 | 1393 | 0.3336 | 0.5047 | 0.5422 | 0.5065 | 0.3564 | 0.2767 | 0.4461 | 0.6732 |
| baseline | 1406 | 0.4125 | 0.6834 | 0.4675 | 0.4888 | 0.3484 | 0.3053 | 0.3086 | 0.6686 |
| qrs_confuse | 1463 | 0.4442 | 0.7408 | 0.7046 | 0.4840 | 0.9563 | 0.6302 | 0.6610 | 0.7105 |
| reset | 1430 | 0.4998 | 0.7603 | 0.7031 | 0.4890 | 0.9699 | 0.6335 | 0.5863 | 0.7105 |
| dropout | 1451 | 0.5013 | 0.8323 | 0.7036 | 0.4981 | 0.9773 | 0.6523 | 0.6177 | 0.6953 |
| rough | 1468 | 0.5538 | 0.7664 | 0.7084 | 0.4967 | 0.9700 | 0.6373 | 0.5413 | 0.7105 |
| highzcr | 1419 | 0.5705 | 0.8187 | 0.7105 | 0.4973 | 0.9739 | 0.6462 | 0.5163 | 0.7105 |
| detail | 1461 | 0.5745 | 0.8186 | 0.7091 | 0.4975 | 0.9726 | 0.6389 | 0.5204 | 0.7105 |
| smooth | 1472 | 0.5818 | 0.8592 | 0.6924 | 0.4906 | 0.9640 | 0.6293 | 0.4718 | 0.7077 |
| identity | 1319 | 0.5937 | 0.8889 | 0.7035 | 0.4897 | 0.9719 | 0.6380 | 0.4721 | 0.7105 |

## Figures

![PCA](E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_waveform_feature_match\ptb_bad_waveform_feature_match_but_bad_test_train_gen12_best_mode_pca.png)

![Waveforms](E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_waveform_feature_match\ptb_bad_waveform_feature_match_but_bad_test_train_gen12_best_mode_waveforms.png)

## Files

- Matched signals: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_waveform_feature_match\ptb_bad_waveform_feature_match_but_bad_test_train_gen12_best_mode_signals.npz`
- Manifest: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_waveform_feature_match\ptb_bad_waveform_feature_match_but_bad_test_train_gen12_best_mode_manifest.csv`
- Feature KS: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_waveform_feature_match\ptb_bad_waveform_feature_match_but_bad_test_train_gen12_best_mode_feature_ks.csv`