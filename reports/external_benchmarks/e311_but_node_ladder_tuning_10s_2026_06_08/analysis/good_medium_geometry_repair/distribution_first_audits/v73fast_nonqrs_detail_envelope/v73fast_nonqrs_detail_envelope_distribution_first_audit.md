# v73fast_nonqrs_detail_envelope Distribution-First Audit

- PTB protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v73fast_nonqrs_detail_envelope\protocol_v73fast_nonqrs_detail_envelope_pc1500_s20260674`
- BUT reference: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\margin_ge_5s_keep_outlier_drop_mediumlike_bad_seed20260623`
- Model decision for the current line: freeze `Event-Factorized SQI Conformer / E1_query_only` as the simple waveform-only Transformer baseline.
- Cross-dataset scores are treated as stress diagnostics only; generation is judged first by visual waveform similarity and feature/PCA distribution overlap.

## Figures

- Shared PCA: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\distribution_first_audits\v73fast_nonqrs_detail_envelope\v73fast_nonqrs_detail_envelope_shared_pca_but_vs_ptb.png`
- Feature CDF overlap: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\distribution_first_audits\v73fast_nonqrs_detail_envelope\v73fast_nonqrs_detail_envelope_key_feature_cdf_overlap.png`
- Representative waveforms: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\distribution_first_audits\v73fast_nonqrs_detail_envelope\v73fast_nonqrs_detail_envelope_representative_waveforms_by_label.png`

## Largest Median Gaps

| class | feature | BUT median | PTB median | PTB-BUT gap (BUT IQR) |
| --- | --- | ---: | ---: | ---: |
| bad | sqi_basSQI | 0.9997 | 0.8232 | -850.463 |
| bad | non_qrs_diff_p95 | 0.3754 | 1.941 | 78.935 |
| bad | qrs_visibility | 0.2442 | 1.038 | 39.739 |
| medium | non_qrs_diff_p95 | 0.09235 | 1.184 | 16.004 |
| good | non_qrs_diff_p95 | 0.05031 | 0.5697 | 13.212 |
| medium | qrs_visibility | 0.2434 | 1.468 | 3.728 |
| bad | amplitude_entropy | 0.8964 | 0.7999 | -2.268 |
| good | detector_agreement | 0.2862 | 0.6667 | 2.095 |
| bad | detector_agreement | 0.5147 | 0.25 | -1.942 |
| bad | baseline_step | 0.0282 | 0.05369 | 1.588 |
| medium | detector_agreement | 0.2979 | 0.5833 | 1.588 |
| good | qrs_visibility | 0.4924 | 1.09 | 1.525 |
| good | amplitude_entropy | 0.6282 | 0.7875 | 1.510 |
| bad | flatline_ratio | 0.007206 | 0.01281 | 1.400 |
| good | flatline_ratio | 0.309 | 0.05524 | -1.215 |
| good | sqi_basSQI | 0.9466 | 0.8364 | -1.041 |
| good | band_30_45 | 0.01725 | 0.006736 | -0.910 |
| bad | band_30_45 | 0.1049 | 0.09256 | -0.802 |

## Interpretation Contract

- Good/medium/bad generation should be accepted only when waveform sheets look plausible and PCA/CDF gaps shrink together.
- If cross-stress disagrees with visual/distribution fit, cross-stress is a domain-shift note, not the optimization target.
- Atlas/KNN/PCA labels remain audit tools; the frozen model inference path remains waveform-derived channels only.