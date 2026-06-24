# v79rawcarrier_bad_envelope_ot_fast_distribution_first Distribution-First Audit

- PTB protocol: `outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v79rawcarrier_bad_envelope_ot_fast\protocol_v79rawcarrier_bad_envelope_ot_fast_pc1500_s20260682`
- BUT reference: `outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\margin_ge_5s_keep_outlier_drop_mediumlike_bad_seed20260623`
- Model decision for the current line: freeze `Event-Factorized SQI Conformer / E1_query_only` as the simple waveform-only Transformer baseline.
- Cross-dataset scores are treated as stress diagnostics only; generation is judged first by visual waveform similarity and feature/PCA distribution overlap.

## Figures

- Shared PCA: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\distribution_first_audits\v79rawcarrier_bad_envelope_ot_fast_distribution_first\v79rawcarrier_bad_envelope_ot_fast_distribution_first_shared_pca_but_vs_ptb.png`
- Feature CDF overlap: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\distribution_first_audits\v79rawcarrier_bad_envelope_ot_fast_distribution_first\v79rawcarrier_bad_envelope_ot_fast_distribution_first_key_feature_cdf_overlap.png`
- Representative waveforms: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\distribution_first_audits\v79rawcarrier_bad_envelope_ot_fast_distribution_first\v79rawcarrier_bad_envelope_ot_fast_distribution_first_representative_waveforms_by_label.png`

## Largest Median Gaps

| class | feature | BUT median | PTB median | PTB-BUT gap (BUT IQR) |
| --- | --- | ---: | ---: | ---: |
| bad | sqi_basSQI | 0.9997 | 0.8132 | -898.822 |
| bad | non_qrs_diff_p95 | 0.3754 | 2.264 | 95.222 |
| bad | qrs_visibility | 0.2442 | 1.055 | 40.602 |
| good | non_qrs_diff_p95 | 0.05031 | 1.308 | 31.984 |
| medium | non_qrs_diff_p95 | 0.09235 | 1.437 | 19.714 |
| medium | qrs_visibility | 0.2434 | 2 | 5.348 |
| good | qrs_visibility | 0.4924 | 2 | 3.848 |
| good | sqi_basSQI | 0.9466 | 0.6247 | -3.041 |
| bad | detector_agreement | 0.5147 | 0.1667 | -2.554 |
| bad | amplitude_entropy | 0.8964 | 0.8065 | -2.114 |
| good | detector_agreement | 0.2862 | 0.6667 | 2.095 |
| bad | baseline_step | 0.0282 | 0.05743 | 1.821 |
| medium | detector_agreement | 0.2979 | 0.5833 | 1.588 |
| good | amplitude_entropy | 0.6282 | 0.7937 | 1.569 |
| medium | sqi_basSQI | 0.9386 | 0.679 | -1.374 |
| bad | flatline_ratio | 0.007206 | 0.01121 | 1.000 |
| good | flatline_ratio | 0.309 | 0.1041 | -0.981 |
| medium | flatline_ratio | 0.1001 | 0.04163 | -0.760 |

## Interpretation Contract

- Good/medium/bad generation should be accepted only when waveform sheets look plausible and PCA/CDF gaps shrink together.
- If cross-stress disagrees with visual/distribution fit, cross-stress is a domain-shift note, not the optimization target.
- Atlas/KNN/PCA labels remain audit tools; the frozen model inference path remains waveform-derived channels only.