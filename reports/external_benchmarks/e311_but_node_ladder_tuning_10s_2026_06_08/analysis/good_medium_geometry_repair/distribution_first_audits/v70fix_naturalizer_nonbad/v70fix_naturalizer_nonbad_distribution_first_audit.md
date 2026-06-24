# v70fix_naturalizer_nonbad Distribution-First Audit

- PTB protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v70fix_naturalizer_nonbad\protocol_v70fix_naturalizer_nonbad_pc1500_s20260672`
- BUT reference: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\margin_ge_5s_keep_outlier_drop_mediumlike_bad_seed20260623`
- Model decision for the current line: freeze `Event-Factorized SQI Conformer / E1_query_only` as the simple waveform-only Transformer baseline.
- Cross-dataset scores are treated as stress diagnostics only; generation is judged first by visual waveform similarity and feature/PCA distribution overlap.

## Figures

- Shared PCA: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\distribution_first_audits\v70fix_naturalizer_nonbad\v70fix_naturalizer_nonbad_shared_pca_but_vs_ptb.png`
- Feature CDF overlap: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\distribution_first_audits\v70fix_naturalizer_nonbad\v70fix_naturalizer_nonbad_key_feature_cdf_overlap.png`
- Representative waveforms: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\distribution_first_audits\v70fix_naturalizer_nonbad\v70fix_naturalizer_nonbad_representative_waveforms_by_label.png`

## Largest Median Gaps

| class | feature | BUT median | PTB median | PTB-BUT gap (BUT IQR) |
| --- | --- | ---: | ---: | ---: |
| bad | sqi_basSQI | 0.9997 | 0.8229 | -851.799 |
| bad | non_qrs_diff_p95 | 0.3754 | 1.944 | 79.116 |
| bad | qrs_visibility | 0.2442 | 1.033 | 39.482 |
| medium | non_qrs_diff_p95 | 0.09235 | 1.414 | 19.383 |
| good | non_qrs_diff_p95 | 0.05031 | 0.7044 | 16.639 |
| medium | qrs_visibility | 0.2434 | 1.928 | 5.129 |
| good | qrs_visibility | 0.4924 | 1.429 | 2.390 |
| bad | amplitude_entropy | 0.8964 | 0.8 | -2.267 |
| good | detector_agreement | 0.2862 | 0.6667 | 2.095 |
| bad | detector_agreement | 0.5147 | 0.25 | -1.942 |
| bad | baseline_step | 0.0282 | 0.05379 | 1.594 |
| medium | detector_agreement | 0.2979 | 0.5833 | 1.588 |
| good | amplitude_entropy | 0.6282 | 0.7914 | 1.547 |
| good | sqi_basSQI | 0.9466 | 0.7912 | -1.468 |
| bad | flatline_ratio | 0.007206 | 0.01281 | 1.400 |
| good | flatline_ratio | 0.309 | 0.07046 | -1.142 |
| good | band_30_45 | 0.01725 | 0.00498 | -1.062 |
| medium | sqi_basSQI | 0.9386 | 0.7451 | -1.024 |

## Interpretation Contract

- Good/medium/bad generation should be accepted only when waveform sheets look plausible and PCA/CDF gaps shrink together.
- If cross-stress disagrees with visual/distribution fit, cross-stress is a domain-shift note, not the optimization target.
- Atlas/KNN/PCA labels remain audit tools; the frozen model inference path remains waveform-derived channels only.