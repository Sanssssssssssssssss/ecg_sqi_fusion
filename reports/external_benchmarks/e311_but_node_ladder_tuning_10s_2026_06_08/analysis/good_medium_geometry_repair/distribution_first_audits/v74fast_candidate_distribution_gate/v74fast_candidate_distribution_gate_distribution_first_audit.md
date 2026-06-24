# v74fast_candidate_distribution_gate Distribution-First Audit

- PTB protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v74fast_candidate_distribution_gate\protocol_v74fast_candidate_distribution_gate_pc1500_s20260675`
- BUT reference: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\margin_ge_5s_keep_outlier_drop_mediumlike_bad_seed20260623`
- Model decision for the current line: freeze `Event-Factorized SQI Conformer / E1_query_only` as the simple waveform-only Transformer baseline.
- Cross-dataset scores are treated as stress diagnostics only; generation is judged first by visual waveform similarity and feature/PCA distribution overlap.

## Figures

- Shared PCA: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\distribution_first_audits\v74fast_candidate_distribution_gate\v74fast_candidate_distribution_gate_shared_pca_but_vs_ptb.png`
- Feature CDF overlap: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\distribution_first_audits\v74fast_candidate_distribution_gate\v74fast_candidate_distribution_gate_key_feature_cdf_overlap.png`
- Representative waveforms: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\distribution_first_audits\v74fast_candidate_distribution_gate\v74fast_candidate_distribution_gate_representative_waveforms_by_label.png`

## Largest Median Gaps

| class | feature | BUT median | PTB median | PTB-BUT gap (BUT IQR) |
| --- | --- | ---: | ---: | ---: |
| bad | sqi_basSQI | 0.9997 | 0.8206 | -863.138 |
| bad | non_qrs_diff_p95 | 0.3754 | 1.975 | 80.645 |
| bad | qrs_visibility | 0.2442 | 1.03 | 39.330 |
| medium | non_qrs_diff_p95 | 0.09235 | 1.169 | 15.783 |
| good | non_qrs_diff_p95 | 0.05031 | 0.5203 | 11.956 |
| medium | qrs_visibility | 0.2434 | 1.468 | 3.728 |
| bad | band_30_45 | 0.1049 | 0.04852 | -3.667 |
| bad | detector_agreement | 0.5147 | 0.08333 | -3.165 |
| bad | amplitude_entropy | 0.8964 | 0.802 | -2.221 |
| good | detector_agreement | 0.2862 | 0.6667 | 2.095 |
| bad | baseline_step | 0.0282 | 0.05466 | 1.648 |
| medium | detector_agreement | 0.2979 | 0.5833 | 1.588 |
| good | qrs_visibility | 0.4924 | 1.09 | 1.526 |
| good | amplitude_entropy | 0.6282 | 0.785 | 1.486 |
| bad | flatline_ratio | 0.007206 | 0.01201 | 1.200 |
| good | flatline_ratio | 0.309 | 0.06325 | -1.176 |
| good | band_30_45 | 0.01725 | 0.004611 | -1.094 |
| good | sqi_basSQI | 0.9466 | 0.8371 | -1.034 |

## Interpretation Contract

- Good/medium/bad generation should be accepted only when waveform sheets look plausible and PCA/CDF gaps shrink together.
- If cross-stress disagrees with visual/distribution fit, cross-stress is a domain-shift note, not the optimization target.
- Atlas/KNN/PCA labels remain audit tools; the frozen model inference path remains waveform-derived channels only.