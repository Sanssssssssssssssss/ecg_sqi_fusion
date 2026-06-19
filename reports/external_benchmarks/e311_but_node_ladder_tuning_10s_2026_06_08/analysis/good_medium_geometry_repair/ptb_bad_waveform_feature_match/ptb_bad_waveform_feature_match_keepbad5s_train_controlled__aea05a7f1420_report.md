# PTB Bad Waveform Feature Match

Generated PTB bad waveforms are selected by recomputing the 39 waveform-computable SQI/morphology primitive features and matching them to BUT bad. This is stronger than sidecar-only 47-feature resampling because the actual waveform changes.

## Summary

- Target: `clean_margin_ge_5s_keep_outlier_bad_train_controlled` rows `4780`
- Base PTB train bad rows: `9229`
- Candidate generated rows: `55374`
- Matched rows: `2200` from `133` unique PTB sources
- Mean KS base -> BUT bad: `0.8022`
- Mean KS matched -> BUT bad: `0.8241`
- Top10 KS base -> BUT bad: `0.9500`
- Top10 KS matched -> BUT bad: `0.9981`
- Selected modes: `{'rightisland_osc': 2129, 'rough': 23, 'dropout': 13, 'reset': 12, 'detail': 8, 'qrs_confuse': 8, 'smooth': 7}`
- Target region counts: `{'right_bad_island': 1843, 'outlier_low_confidence': 310, 'near_bad_boundary': 47}`

## Remaining Largest Matched KS

| feature | matched KS | target median | matched median |
| --- | ---: | ---: | ---: |
| hjorth_mobility | 1.0000 | 1.4336 | 1.1596 |
| higuchi_fd_proxy | 1.0000 | 1.0273 | 0.9885 |
| diff_zero_crossing_rate | 1.0000 | 0.7115 | 0.4191 |
| zero_crossing_rate | 1.0000 | 0.5741 | 0.4147 |
| sqi_bSQI | 1.0000 | 0.1096 | 0.9020 |
| band_15_30 | 0.9991 | 0.4573 | 0.8672 |
| diff_abs_p95 | 0.9971 | 2.7757 | 1.8682 |
| non_qrs_diff_p95 | 0.9971 | 2.7757 | 1.8682 |
| fatal_or_score | 0.9971 | 1.1354 | 0.7834 |
| hjorth_complexity | 0.9903 | 1.2147 | 1.0463 |
| wavelet_e4 | 0.9750 | 0.1042 | 0.0079 |
| sqi_fSQI | 0.9750 | 0.8891 | 0.9908 |

## Candidate Mode KS Summary

| mode | rows | mean KS | top10 KS | zcr KS | diff-zcr KS | band15-30 KS | non-qrs-diff KS | bSQI KS | flatline KS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 4167 | 0.7697 | 0.9877 | 1.0000 | 0.9995 | 0.9210 | 0.9621 | 0.9833 | 0.9744 |
| flatstep_qrsvisible | 4161 | 0.7808 | 0.9995 | 1.0000 | 0.9995 | 0.9974 | 0.9935 | 0.9427 | 0.9994 |
| detail | 4197 | 0.7959 | 0.9522 | 1.0000 | 1.0000 | 0.8302 | 0.7756 | 0.8822 | 0.9133 |
| smooth | 4200 | 0.7997 | 0.9559 | 1.0000 | 0.9990 | 0.8314 | 0.8669 | 0.9187 | 0.9169 |
| identity | 9229 | 0.8006 | 0.9489 | 0.9999 | 0.9998 | 0.8349 | 0.7570 | 0.8420 | 0.9213 |
| rough | 4256 | 0.8016 | 0.9615 | 1.0000 | 1.0000 | 0.8367 | 0.7820 | 0.8911 | 0.9152 |
| mixed | 4182 | 0.8048 | 0.9959 | 1.0000 | 1.0000 | 0.9548 | 0.9886 | 0.9866 | 0.9939 |
| rightisland_osc | 4193 | 0.8070 | 1.0000 | 1.0000 | 1.0000 | 0.9971 | 1.0000 | 1.0000 | 0.6540 |
| reset | 4196 | 0.8095 | 0.9615 | 0.9998 | 0.9998 | 0.8412 | 0.7443 | 0.8419 | 0.9211 |
| qrs_confuse | 4237 | 0.8168 | 0.9690 | 1.0000 | 1.0000 | 0.8347 | 0.7834 | 0.8617 | 0.9224 |
| record111 | 4200 | 0.8172 | 0.9929 | 1.0000 | 0.9998 | 0.9401 | 0.9741 | 0.9836 | 0.9958 |
| dropout | 4156 | 0.8202 | 0.9704 | 1.0000 | 0.9987 | 0.8346 | 0.7348 | 0.7627 | 0.9674 |

## Figures

![PCA](E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_waveform_feature_match\ptb_bad_waveform_feature_match_keepbad5s_train_controlled__aea05a7f1420_pca.png)

![Waveforms](E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_waveform_feature_match\ptb_bad_waveform_feature_match_keepbad5s_train_controlled__aea05a7f1420_waveforms.png)

## Files

- Matched signals: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_waveform_feature_match\ptb_bad_waveform_feature_match_keepbad5s_train_controlled__aea05a7f1420_signals.npz`
- Manifest: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_waveform_feature_match\ptb_bad_waveform_feature_match_keepbad5s_train_controlled__aea05a7f1420_manifest.csv`
- Feature KS: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_waveform_feature_match\ptb_bad_waveform_feature_match_keepbad5s_train_controlled__aea05a7f1420_feature_ks.csv`