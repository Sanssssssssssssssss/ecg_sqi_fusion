# PTB Bad 47-Feature Distribution Match

This artifact matches PTB synthetic bad rows to the BUT bad distribution using the existing 47-column SQI/geometry schema. BUT rows are target-distribution diagnostics only; output signals remain PTB synthetic waveforms.

## Summary

- Target: `but_bad_all` rows `5285`
- Candidate: PTB synthetic bad `train` rows `1319`
- Match space: `all47` columns `47`
- Matched rows: `1816` using `708` unique PTB bad sources
- Mean KS base -> BUT bad: `0.0957`
- Mean KS matched -> BUT bad: `0.0759`
- Top10 KS base -> BUT bad: `0.1867`
- Top10 KS matched -> BUT bad: `0.1260`
- Match distance q50/q90: `0.4414` / `10.6837`

## Remaining Largest Matched KS

| feature | matched KS | target median | matched median |
| --- | ---: | ---: | ---: |
| region_confidence | 0.2172 | 0.6741 | 0.7684 |
| boundary_confidence | 0.2025 | 0.6749 | 0.7739 |
| pca_margin | 0.1626 | 10.8395 | 10.9302 |
| qrs_visibility | 0.1440 | 0.2460 | 0.2487 |
| sqi_sSQI | 0.0968 | -0.0037 | 0.0119 |
| qrs_band_ratio | 0.0912 | 0.8091 | 0.8102 |
| pc2 | 0.0881 | -0.0339 | -0.0446 |
| sqi_fSQI | 0.0873 | 0.0000 | 0.0000 |
| pc3 | 0.0853 | -1.1134 | -1.0857 |
| diff_abs_p95 | 0.0852 | 0.3920 | 0.3909 |
| sqi_bSQI | 0.0826 | 0.0000 | 0.0000 |
| wavelet_e3 | 0.0814 | 0.2607 | 0.2622 |

## Figures

![PCA](E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_47_feature_match\ptb_bad_47match_but_bad_all_train_all47_pca.png)

![Waveforms](E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_47_feature_match\ptb_bad_47match_but_bad_all_train_all47_waveforms.png)

## Files

- Manifest: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_47_feature_match\ptb_bad_47match_but_bad_all_train_all47_manifest.csv`
- Feature KS: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_47_feature_match\ptb_bad_47match_but_bad_all_train_all47_feature_ks.csv`
- Matched PTB signals: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_47_feature_match\ptb_bad_47match_but_bad_all_train_all47_signals.npz`