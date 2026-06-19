# PTB Bad 47-Feature Distribution Match

This artifact matches PTB synthetic bad rows to the BUT bad distribution using the existing 47-column SQI/geometry schema. BUT rows are target-distribution diagnostics only; output signals remain PTB synthetic waveforms.

## Summary

- Target: `but_bad_all` rows `5285`
- Candidate: PTB synthetic bad `train` rows `1319`
- Match space: `primitive39` columns `39`
- Matched rows: `1816` using `726` unique PTB bad sources
- Mean KS base -> BUT bad: `0.0749`
- Mean KS matched -> BUT bad: `0.0665`
- Top10 KS base -> BUT bad: `0.1072`
- Top10 KS matched -> BUT bad: `0.0910`
- Match distance q50/q90: `0.4377` / `11.6273`

## Remaining Largest Matched KS

| feature | matched KS | target median | matched median |
| --- | ---: | ---: | ---: |
| qrs_visibility | 0.1284 | 0.2460 | 0.2486 |
| detector_agreement | 0.1104 | 0.4450 | 0.4450 |
| sqi_fSQI | 0.0994 | 0.0000 | 0.0000 |
| qrs_band_ratio | 0.0912 | 0.8091 | 0.8101 |
| sqi_bSQI | 0.0831 | 0.0000 | 0.0000 |
| wavelet_e3 | 0.0814 | 0.2607 | 0.2620 |
| diff_abs_p95 | 0.0805 | 0.3920 | 0.3908 |
| wavelet_e2 | 0.0795 | 0.4054 | 0.4071 |
| wavelet_e1 | 0.0780 | 0.0354 | 0.0358 |
| sqi_sSQI | 0.0778 | -0.0037 | 0.0093 |
| template_corr | 0.0763 | 0.1644 | 0.1575 |
| band_30_45 | 0.0737 | 0.1034 | 0.1031 |

## Figures

![PCA](E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_47_feature_match\ptb_bad_47match_but_bad_all_train_primitive39_pca.png)

![Waveforms](E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_47_feature_match\ptb_bad_47match_but_bad_all_train_primitive39_waveforms.png)

## Files

- Manifest: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_47_feature_match\ptb_bad_47match_but_bad_all_train_primitive39_manifest.csv`
- Feature KS: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_47_feature_match\ptb_bad_47match_but_bad_all_train_primitive39_feature_ks.csv`
- Matched PTB signals: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_47_feature_match\ptb_bad_47match_but_bad_all_train_primitive39_signals.npz`