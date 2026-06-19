# PTB Bad Waveform Feature Match

Generated PTB bad waveforms are selected by recomputing the 39 waveform-computable SQI/morphology primitive features and matching them to BUT bad. This is stronger than sidecar-only 47-feature resampling because the actual waveform changes.

## Summary

- Target: `clean_margin_ge_5s_drop_outlier_bad_test` rows `118`
- Base PTB train bad rows: `1319`
- Candidate generated rows: `27699`
- Matched rows: `1450` from `8` unique PTB sources
- Mean KS base -> BUT bad: `0.6900`
- Mean KS matched -> BUT bad: `0.5524`
- Top10 KS base -> BUT bad: `0.9956`
- Top10 KS matched -> BUT bad: `0.8374`
- Selected modes: `{'cleanhf_qrsdrop': 633, 'dropout': 378, 'cleanhf': 289, 'highzcr': 150}`
- Target region counts: `{'near_bad_boundary': 1450}`

## Remaining Largest Matched KS

| feature | matched KS | target median | matched median |
| --- | ---: | ---: | ---: |
| band_15_30 | 1.0000 | 0.2231 | 0.3643 |
| sqi_bSQI | 1.0000 | 0.0753 | 0.3571 |
| wavelet_e3 | 0.9924 | 0.3219 | 0.4838 |
| wavelet_e2 | 0.9915 | 0.2622 | 0.0465 |
| wavelet_e1 | 0.9492 | 0.1098 | 0.0212 |
| diff_zero_crossing_rate | 0.8955 | 0.6755 | 0.6234 |
| wavelet_e0 | 0.6376 | 0.0173 | 0.0028 |
| hjorth_complexity | 0.6359 | 1.2466 | 1.1127 |
| sample_entropy_proxy | 0.6359 | 0.8518 | 0.7556 |
| sqi_sSQI | 0.6359 | 0.1482 | 0.2444 |
| non_qrs_rms_ratio | 0.6359 | 0.9579 | 0.9806 |
| zero_crossing_rate | 0.6290 | 0.4912 | 0.5404 |

## Candidate Mode KS Summary

| mode | rows | mean KS | top10 KS | zcr KS | diff-zcr KS | band15-30 KS | non-qrs-diff KS | bSQI KS | flatline KS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cleanhf | 1908 | 0.5782 | 0.9781 | 0.9182 | 0.3864 | 0.7231 | 0.8477 | 1.0000 | 0.2523 |
| cleanhf_qrsdrop | 1872 | 0.5900 | 0.9829 | 0.9563 | 0.3498 | 0.7763 | 0.8875 | 0.9995 | 0.2547 |
| rough | 1861 | 0.6000 | 0.9947 | 1.0000 | 1.0000 | 0.9710 | 0.9883 | 0.9919 | 0.4728 |
| reset | 1911 | 0.6051 | 0.9945 | 1.0000 | 1.0000 | 0.9702 | 0.9852 | 0.8455 | 0.4648 |
| detail | 1882 | 0.6411 | 0.9937 | 1.0000 | 1.0000 | 0.9745 | 0.9830 | 0.9968 | 0.4647 |
| highzcr | 1912 | 0.6429 | 0.9941 | 1.0000 | 1.0000 | 0.9702 | 0.9852 | 0.9927 | 0.4612 |
| qrs_confuse | 1956 | 0.6599 | 0.9963 | 1.0000 | 1.0000 | 0.9545 | 0.9908 | 0.7203 | 0.5075 |
| smooth | 1862 | 0.6794 | 0.9936 | 1.0000 | 0.9995 | 0.9629 | 0.9835 | 0.9946 | 0.4531 |
| identity | 1319 | 0.7028 | 0.9964 | 1.0000 | 1.0000 | 0.9719 | 0.9832 | 0.9917 | 0.4781 |
| dropout | 1833 | 0.7293 | 0.9864 | 0.9995 | 0.9973 | 0.9727 | 0.4983 | 0.8025 | 0.7743 |
| baseline | 1855 | 0.7421 | 1.0000 | 1.0000 | 0.9995 | 0.5782 | 1.0000 | 0.9914 | 0.9035 |
| mixed | 1885 | 0.7596 | 0.9990 | 1.0000 | 1.0000 | 0.7528 | 0.9830 | 0.9082 | 0.9851 |

## Figures

![PCA](E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_waveform_feature_match\ptb_bad_waveform_feature_match_cleanbad5s_test_gen20_nearest_pca.png)

![Waveforms](E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_waveform_feature_match\ptb_bad_waveform_feature_match_cleanbad5s_test_gen20_nearest_waveforms.png)

## Files

- Matched signals: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_waveform_feature_match\ptb_bad_waveform_feature_match_cleanbad5s_test_gen20_nearest_signals.npz`
- Manifest: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_waveform_feature_match\ptb_bad_waveform_feature_match_cleanbad5s_test_gen20_nearest_manifest.csv`
- Feature KS: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_waveform_feature_match\ptb_bad_waveform_feature_match_cleanbad5s_test_gen20_nearest_feature_ks.csv`