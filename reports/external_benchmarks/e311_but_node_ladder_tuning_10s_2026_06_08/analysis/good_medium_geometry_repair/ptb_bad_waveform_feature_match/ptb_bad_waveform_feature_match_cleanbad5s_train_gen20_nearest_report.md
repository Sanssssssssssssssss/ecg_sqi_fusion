# PTB Bad Waveform Feature Match

Generated PTB bad waveforms are selected by recomputing the 39 waveform-computable SQI/morphology primitive features and matching them to BUT bad. This is stronger than sidecar-only 47-feature resampling because the actual waveform changes.

## Summary

- Target: `clean_margin_ge_5s_drop_outlier_bad_train` rows `3963`
- Base PTB train bad rows: `1319`
- Candidate generated rows: `27699`
- Matched rows: `1450` from `3` unique PTB sources
- Mean KS base -> BUT bad: `0.6897`
- Mean KS matched -> BUT bad: `0.8890`
- Top10 KS base -> BUT bad: `0.9922`
- Top10 KS matched -> BUT bad: `0.9986`
- Selected modes: `{'cleanhf': 1448, 'cleanhf_qrsdrop': 2}`
- Target region counts: `{'right_bad_island': 1450}`

## Remaining Largest Matched KS

| feature | matched KS | target median | matched median |
| --- | ---: | ---: | ---: |
| wavelet_e0 | 1.0000 | 0.0002 | 0.0025 |
| sqi_bSQI | 1.0000 | 0.0910 | 0.4955 |
| non_qrs_rms_ratio | 1.0000 | 1.0345 | 1.0116 |
| wavelet_e1 | 1.0000 | 0.0105 | 0.0249 |
| zero_crossing_rate | 0.9986 | 0.5749 | 0.4940 |
| higuchi_fd_proxy | 0.9981 | 1.0274 | 1.0107 |
| std | 0.9974 | 1.0983 | 0.9880 |
| hjorth_activity | 0.9974 | 1.2062 | 0.9761 |
| rms | 0.9974 | 1.0984 | 0.9880 |
| diff_zero_crossing_rate | 0.9972 | 0.7123 | 0.6458 |
| mean_abs | 0.9972 | 0.8583 | 0.7937 |
| amplitude_entropy | 0.9970 | 0.8267 | 0.8015 |

## Candidate Mode KS Summary

| mode | rows | mean KS | top10 KS | zcr KS | diff-zcr KS | band15-30 KS | non-qrs-diff KS | bSQI KS | flatline KS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dropout | 1702 | 0.6580 | 0.9670 | 1.0000 | 1.0000 | 0.9783 | 0.5773 | 0.7368 | 0.8779 |
| reset | 1779 | 0.6810 | 0.9901 | 1.0000 | 1.0000 | 0.9386 | 0.9927 | 0.8298 | 0.7182 |
| highzcr | 1780 | 0.6975 | 0.9947 | 1.0000 | 1.0000 | 0.9713 | 0.9921 | 0.9944 | 0.7312 |
| qrs_confuse | 1736 | 0.6980 | 0.9988 | 1.0000 | 1.0000 | 0.7696 | 0.9980 | 0.6699 | 0.7639 |
| detail | 1734 | 0.6998 | 0.9948 | 1.0000 | 1.0000 | 0.9735 | 0.9920 | 0.9969 | 0.7159 |
| identity | 1319 | 0.7034 | 0.9943 | 1.0000 | 1.0000 | 0.9712 | 0.9914 | 0.9917 | 0.7243 |
| smooth | 1743 | 0.7037 | 0.9943 | 1.0000 | 0.9995 | 0.9742 | 0.9902 | 0.9960 | 0.7154 |
| rough | 1800 | 0.7068 | 0.9976 | 1.0000 | 1.0000 | 0.9606 | 0.9979 | 0.9912 | 0.7415 |
| mixed | 1793 | 0.7186 | 0.9985 | 1.0000 | 1.0000 | 0.9500 | 0.9908 | 0.8884 | 0.9891 |
| cleanhf | 1774 | 0.7223 | 0.9941 | 0.5495 | 0.8190 | 0.9737 | 0.7820 | 1.0000 | 0.1956 |
| cleanhf_qrsdrop | 1762 | 0.7236 | 0.9937 | 0.6063 | 0.7979 | 0.9770 | 0.8155 | 0.9989 | 0.2054 |
| record111 | 1717 | 0.7304 | 0.9976 | 1.0000 | 1.0000 | 0.9070 | 0.9843 | 0.8837 | 0.9852 |

## Figures

![PCA](E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_waveform_feature_match\ptb_bad_waveform_feature_match_cleanbad5s_train_gen20_nearest_pca.png)

![Waveforms](E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_waveform_feature_match\ptb_bad_waveform_feature_match_cleanbad5s_train_gen20_nearest_waveforms.png)

## Files

- Matched signals: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_waveform_feature_match\ptb_bad_waveform_feature_match_cleanbad5s_train_gen20_nearest_signals.npz`
- Manifest: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_waveform_feature_match\ptb_bad_waveform_feature_match_cleanbad5s_train_gen20_nearest_manifest.csv`
- Feature KS: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_waveform_feature_match\ptb_bad_waveform_feature_match_cleanbad5s_train_gen20_nearest_feature_ks.csv`