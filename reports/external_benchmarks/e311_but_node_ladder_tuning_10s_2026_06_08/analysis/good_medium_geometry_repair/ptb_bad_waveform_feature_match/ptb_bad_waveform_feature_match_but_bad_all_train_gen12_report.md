# PTB Bad Waveform Feature Match

Generated PTB bad waveforms are selected by recomputing the 39 waveform-computable SQI/morphology primitive features and matching them to BUT bad. This is stronger than sidecar-only 47-feature resampling because the actual waveform changes.

## Summary

- Target: `but_bad_all` rows `5285`
- Base PTB train bad rows: `1319`
- Candidate generated rows: `17147`
- Matched rows: `1816` from `93` unique PTB sources
- Mean KS base -> BUT bad: `0.6439`
- Mean KS matched -> BUT bad: `0.8275`
- Top10 KS base -> BUT bad: `0.9328`
- Top10 KS matched -> BUT bad: `0.9223`
- Selected modes: `{'dropout': 1630, 'smooth': 39, 'stress': 29, 'record111': 28, 'mixed': 26, 'highzcr': 23, 'detail': 17, 'rough': 6, 'identity': 6, 'qrs_confuse': 6, 'reset': 5, 'baseline': 1}`
- Target region counts: `{'right_bad_island': 1388, 'outlier_low_confidence': 387, 'near_bad_boundary': 41}`

## Remaining Largest Matched KS

| feature | matched KS | target median | matched median |
| --- | ---: | ---: | ---: |
| diff_zero_crossing_rate | 0.9336 | 0.7107 | 0.4439 |
| higuchi_fd_proxy | 0.9330 | 1.0270 | 0.9788 |
| hjorth_mobility | 0.9330 | 1.4326 | 1.0073 |
| zero_crossing_rate | 0.9330 | 0.5733 | 0.3339 |
| band_15_30 | 0.9254 | 0.4557 | 0.8506 |
| fatal_or_score | 0.9200 | 1.1318 | 0.8127 |
| diff_abs_p95 | 0.9177 | 2.7663 | 1.9912 |
| non_qrs_diff_p95 | 0.9177 | 2.7663 | 1.9912 |
| sqi_bSQI | 0.9055 | 0.3572 | 0.4492 |
| sqi_fSQI | 0.9039 | 0.8881 | 0.9453 |
| flatline_ratio | 0.9025 | 0.0064 | 0.0136 |
| rms | 0.9017 | 1.0983 | 1.0132 |

## Candidate Mode KS Summary

| mode | rows | mean KS | top10 KS | zcr KS | diff-zcr KS | band15-30 KS | non-qrs-diff KS | bSQI KS | flatline KS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dropout | 1451 | 0.6016 | 0.9027 | 0.9328 | 0.9336 | 0.9759 | 0.4714 | 0.4258 | 0.8105 |
| reset | 1430 | 0.6123 | 0.9179 | 0.9330 | 0.9318 | 0.9390 | 0.9118 | 0.2871 | 0.6653 |
| qrs_confuse | 1463 | 0.6293 | 0.9165 | 0.9332 | 0.9336 | 0.7533 | 0.9183 | 0.4721 | 0.6867 |
| highzcr | 1419 | 0.6317 | 0.9197 | 0.9330 | 0.9336 | 0.9679 | 0.9108 | 0.3969 | 0.6538 |
| detail | 1461 | 0.6322 | 0.9210 | 0.9330 | 0.9336 | 0.9695 | 0.9140 | 0.4051 | 0.6652 |
| smooth | 1472 | 0.6339 | 0.9195 | 0.9328 | 0.9315 | 0.9640 | 0.9063 | 0.4191 | 0.6546 |
| rough | 1468 | 0.6366 | 0.9228 | 0.9332 | 0.9336 | 0.9609 | 0.9164 | 0.3789 | 0.6825 |
| identity | 1319 | 0.6413 | 0.9328 | 0.9330 | 0.9336 | 0.9712 | 0.9105 | 0.4388 | 0.6609 |
| mixed | 1433 | 0.6459 | 0.9263 | 0.9336 | 0.9336 | 0.8387 | 0.9072 | 0.5373 | 0.9156 |
| record111 | 1393 | 0.6566 | 0.9248 | 0.9334 | 0.9336 | 0.7993 | 0.9052 | 0.4715 | 0.9182 |
| stress | 1432 | 0.6591 | 0.9198 | 0.9336 | 0.9336 | 0.7834 | 0.8584 | 0.4159 | 0.9149 |
| baseline | 1406 | 0.6899 | 0.9286 | 0.9334 | 0.9336 | 0.7254 | 0.9211 | 0.8219 | 0.8816 |

## Figures

![PCA](E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_waveform_feature_match\ptb_bad_waveform_feature_match_but_bad_all_train_gen12_pca.png)

![Waveforms](E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_waveform_feature_match\ptb_bad_waveform_feature_match_but_bad_all_train_gen12_waveforms.png)

## Files

- Matched signals: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_waveform_feature_match\ptb_bad_waveform_feature_match_but_bad_all_train_gen12_signals.npz`
- Manifest: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_waveform_feature_match\ptb_bad_waveform_feature_match_but_bad_all_train_gen12_manifest.csv`
- Feature KS: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_waveform_feature_match\ptb_bad_waveform_feature_match_but_bad_all_train_gen12_feature_ks.csv`