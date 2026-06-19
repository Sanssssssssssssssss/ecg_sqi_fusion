# PTB Bad Distribution Match

External-only distribution diagnostic/generator.  It builds a PTB bad bank selected in waveform primitive space to match BUT bad target distribution.

## Summary

- Target: `but_bad_all`
- BUT target bad rows: `5285`
- Base PTB bad rows: `1816`
- Candidate PTB bad rows: `23608`
- Selected matched rows: `1816`
- Mean KS base -> BUT bad: `0.5702`
- Mean KS matched -> BUT bad: `0.5158`
- Top10 KS base -> BUT bad: `0.9144`
- Top10 KS matched -> BUT bad: `0.8322`
- Selected modes: `{'smooth': 342, 'highzcr': 290, 'detail': 255, 'identity': 237, 'reset': 230, 'dropout': 200, 'rough': 177, 'stress': 39, 'mixed': 35, 'baseline': 11}`

## Figures

![PCA](E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_distribution_match\ptb_bad_distribution_match_pca.png)

![Waveforms](E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_distribution_match\ptb_bad_distribution_match_waveforms.png)

## Files

- Matched signals: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_distribution_match\ptb_bad_distribution_matched_signals.npz`
- Manifest: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_distribution_match\ptb_bad_distribution_matched_manifest.csv`
- Feature KS: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\ptb_bad_distribution_match\ptb_bad_distribution_match_feature_ks.csv`
