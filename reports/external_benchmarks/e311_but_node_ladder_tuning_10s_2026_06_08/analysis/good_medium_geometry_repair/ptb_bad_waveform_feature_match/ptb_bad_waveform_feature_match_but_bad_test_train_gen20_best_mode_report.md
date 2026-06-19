# PTB Bad Waveform Feature Match

Generated PTB bad waveforms are selected by recomputing the 39 waveform-computable SQI/morphology primitive features and matching them to BUT bad. This is stronger than sidecar-only 47-feature resampling because the actual waveform changes.

## Summary

- Target: `but_bad_test` rows `411`
- Base PTB train bad rows: `1319`
- Candidate generated rows: `27699`
- Matched rows: `1450` from `911` unique PTB sources
- Mean KS base -> BUT bad: `0.5975`
- Mean KS matched -> BUT bad: `0.3226`
- Top10 KS base -> BUT bad: `0.8889`
- Top10 KS matched -> BUT bad: `0.5116`
- Selected modes: `{'mixed': 1450}`
- Target region counts: `{'outlier_low_confidence': 1012, 'near_bad_boundary': 438}`

## Remaining Largest Matched KS

| feature | matched KS | target median | matched median |
| --- | ---: | ---: | ---: |
| flatline_ratio | 0.6399 | 0.2458 | 0.0432 |
| sqi_sSQI | 0.5528 | 0.3760 | 0.6198 |
| sample_entropy_proxy | 0.5528 | 0.6240 | 0.3802 |
| wavelet_e2 | 0.5390 | 0.1854 | 0.0637 |
| sqi_fSQI | 0.5275 | 0.7576 | 0.8649 |
| diff_zero_crossing_rate | 0.5124 | 0.3510 | 0.4247 |
| baseline_step | 0.4562 | 0.0727 | 0.0482 |
| sqi_basSQI | 0.4562 | 0.7747 | 0.8384 |
| zero_crossing_rate | 0.4484 | 0.0440 | 0.1273 |
| wavelet_e1 | 0.4312 | 0.2817 | 0.1785 |
| wavelet_e3 | 0.4140 | 0.0582 | 0.2072 |
| wavelet_e4 | 0.3513 | 0.0153 | 0.0247 |

## Candidate Mode KS Summary

| mode | rows | mean KS | top10 KS | zcr KS | diff-zcr KS | band15-30 KS | non-qrs-diff KS | bSQI KS | flatline KS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mixed | 2243 | 0.3203 | 0.5159 | 0.4643 | 0.5118 | 0.2292 | 0.2777 | 0.3236 | 0.6432 |
| stress | 2193 | 0.3221 | 0.5168 | 0.5533 | 0.5113 | 0.3457 | 0.2376 | 0.4002 | 0.6723 |
| record111 | 2175 | 0.3341 | 0.5112 | 0.5272 | 0.5048 | 0.3491 | 0.2711 | 0.3687 | 0.6711 |
| baseline | 2218 | 0.4087 | 0.6844 | 0.4380 | 0.4925 | 0.3313 | 0.3039 | 0.1938 | 0.6648 |
| qrs_confuse | 2132 | 0.4416 | 0.7354 | 0.7016 | 0.4844 | 0.9498 | 0.6264 | 0.6047 | 0.7105 |
| flatstep_qrsvisible | 2162 | 0.4915 | 0.7334 | 0.2895 | 0.5233 | 0.4837 | 0.4722 | 0.2892 | 0.2891 |
| dropout | 2285 | 0.4918 | 0.8230 | 0.7052 | 0.4978 | 0.9681 | 0.6484 | 0.5433 | 0.6933 |
| reset | 2215 | 0.4986 | 0.7597 | 0.7062 | 0.4903 | 0.9666 | 0.6347 | 0.5134 | 0.7105 |
| rough | 2208 | 0.5485 | 0.7624 | 0.7096 | 0.4971 | 0.9697 | 0.6324 | 0.4858 | 0.7105 |
| highzcr | 2131 | 0.5710 | 0.8197 | 0.7100 | 0.4996 | 0.9751 | 0.6461 | 0.4744 | 0.7105 |
| detail | 2181 | 0.5746 | 0.8224 | 0.7100 | 0.5004 | 0.9771 | 0.6460 | 0.4631 | 0.7105 |
| smooth | 2237 | 0.5874 | 0.8666 | 0.7000 | 0.4908 | 0.9745 | 0.6416 | 0.4283 | 0.7100 |

## Figures

![PCA](E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_waveform_feature_match\ptb_bad_waveform_feature_match_but_bad_test_train_gen20_best_mode_pca.png)

![Waveforms](E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_waveform_feature_match\ptb_bad_waveform_feature_match_but_bad_test_train_gen20_best_mode_waveforms.png)

## Files

- Matched signals: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_waveform_feature_match\ptb_bad_waveform_feature_match_but_bad_test_train_gen20_best_mode_signals.npz`
- Manifest: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_waveform_feature_match\ptb_bad_waveform_feature_match_but_bad_test_train_gen20_best_mode_manifest.csv`
- Feature KS: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_waveform_feature_match\ptb_bad_waveform_feature_match_but_bad_test_train_gen20_best_mode_feature_ks.csv`