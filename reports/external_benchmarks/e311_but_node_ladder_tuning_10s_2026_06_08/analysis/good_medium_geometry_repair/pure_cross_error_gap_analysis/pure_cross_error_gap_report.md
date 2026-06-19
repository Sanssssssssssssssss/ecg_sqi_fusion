# Pure-Cross Error Gap Analysis

Strict read-only analysis for PTB-only -> clean-BUT and clean-BUT-only -> PTB.
Joint/replay and rule artifacts are intentionally excluded.

| Target | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_train_to_clean_but_test | 0.929040 | 0.934530 | 0.939243 | 0.920077 | 1.000000 | 61 | 157 | 0 |
| clean_but_train_to_ptb_test_ensemble | 0.909277 | 0.909990 | 0.910042 | 0.895292 | 0.979253 | 43 | 112 | 5 |

## Outputs

### ptb_train_to_clean_but_test
- Rows: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\pure_cross_error_gap_analysis\ptb_train_to_clean_but_test_rows.csv`
- Feature gaps: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\pure_cross_error_gap_analysis\ptb_train_to_clean_but_test_feature_gaps.csv`
- Waveforms: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\pure_cross_error_gap_analysis\ptb_train_to_clean_but_test_waveform_errors.png`

### clean_but_train_to_ptb_test_ensemble
- Rows: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\pure_cross_error_gap_analysis\clean_but_train_to_ptb_test_ensemble_rows.csv`
- Feature gaps: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\pure_cross_error_gap_analysis\clean_but_train_to_ptb_test_ensemble_feature_gaps.csv`
- Waveforms: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\pure_cross_error_gap_analysis\clean_but_train_to_ptb_test_ensemble_waveform_errors.png`

Interpretation should focus on stable feature gaps, not on one-off split luck.