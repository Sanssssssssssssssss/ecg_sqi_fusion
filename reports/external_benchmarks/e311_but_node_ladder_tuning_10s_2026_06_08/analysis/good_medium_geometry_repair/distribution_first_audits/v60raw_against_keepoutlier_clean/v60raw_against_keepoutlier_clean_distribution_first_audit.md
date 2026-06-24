# v60raw_against_keepoutlier_clean Distribution-First Audit

- PTB protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v60raw\protocol_v60raw_pc3000_s20260650`
- BUT reference: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\margin_ge_5s_keep_outlier_drop_mediumlike_bad_seed20260623`
- Model decision for the current line: freeze `Event-Factorized SQI Conformer / E1_query_only` as the simple waveform-only Transformer baseline.
- Cross-dataset scores are treated as stress diagnostics only; generation is judged first by visual waveform similarity and feature/PCA distribution overlap.

## Figures

- Shared PCA: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\distribution_first_audits\v60raw_against_keepoutlier_clean\v60raw_against_keepoutlier_clean_shared_pca_but_vs_ptb.png`
- Feature CDF overlap: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\distribution_first_audits\v60raw_against_keepoutlier_clean\v60raw_against_keepoutlier_clean_key_feature_cdf_overlap.png`
- Representative waveforms: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\distribution_first_audits\v60raw_against_keepoutlier_clean\v60raw_against_keepoutlier_clean_representative_waveforms_by_label.png`

## Largest Median Gaps

| class | feature | BUT median | PTB median | PTB-BUT gap (BUT IQR) |
| --- | --- | ---: | ---: | ---: |
| bad | sqi_basSQI | 0.9997 | 0.8246 | -843.914 |
| bad | non_qrs_diff_p95 | 0.3754 | 1.936 | 78.690 |
| bad | qrs_visibility | 0.2442 | 1.029 | 39.283 |
| good | non_qrs_diff_p95 | 0.05031 | 1.34 | 32.805 |
| medium | non_qrs_diff_p95 | 0.09235 | 1.82 | 25.333 |
| medium | qrs_visibility | 0.2434 | 2 | 5.348 |
| good | qrs_visibility | 0.4924 | 2 | 3.848 |
| good | sqi_basSQI | 0.9466 | 0.6104 | -3.176 |
| bad | amplitude_entropy | 0.8964 | 0.7985 | -2.303 |
| bad | detector_agreement | 0.5147 | 0.25 | -1.942 |
| good | amplitude_entropy | 0.6282 | 0.808 | 1.705 |
| good | detector_agreement | 0.2862 | 0.5833 | 1.636 |
| bad | flatline_ratio | 0.007206 | 0.01361 | 1.600 |
| medium | detector_agreement | 0.2979 | 0.5833 | 1.588 |
| bad | baseline_step | 0.0282 | 0.05319 | 1.556 |
| medium | sqi_basSQI | 0.9386 | 0.6853 | -1.341 |
| bad | band_30_45 | 0.1049 | 0.09056 | -0.932 |
| good | flatline_ratio | 0.309 | 0.1281 | -0.866 |

## Interpretation Contract

- Good/medium/bad generation should be accepted only when waveform sheets look plausible and PCA/CDF gaps shrink together.
- If cross-stress disagrees with visual/distribution fit, cross-stress is a domain-shift note, not the optimization target.
- Atlas/KNN/PCA labels remain audit tools; the frozen model inference path remains waveform-derived channels only.