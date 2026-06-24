# raw_ptbxl_carrier_max9000 Distribution-First Audit

- PTB protocol: `outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\raw_ptbxl_carrier_protocols\raw_ptbxl_lead2_clean_noise_notes_max9000_seed20260679`
- BUT reference: `outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\margin_ge_5s_keep_outlier_drop_mediumlike_bad_seed20260623`
- Model decision for the current line: freeze `Event-Factorized SQI Conformer / E1_query_only` as the simple waveform-only Transformer baseline.
- Cross-dataset scores are treated as stress diagnostics only; generation is judged first by visual waveform similarity and feature/PCA distribution overlap.

## Figures

- Shared PCA: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\distribution_first_audits\raw_ptbxl_carrier_max9000\raw_ptbxl_carrier_max9000_shared_pca_but_vs_ptb.png`
- Feature CDF overlap: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\distribution_first_audits\raw_ptbxl_carrier_max9000\raw_ptbxl_carrier_max9000_key_feature_cdf_overlap.png`
- Representative waveforms: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\distribution_first_audits\raw_ptbxl_carrier_max9000\raw_ptbxl_carrier_max9000_representative_waveforms_by_label.png`

## Largest Median Gaps

| class | feature | BUT median | PTB median | PTB-BUT gap (BUT IQR) |
| --- | --- | ---: | ---: | ---: |
| bad | sqi_basSQI | 0.9997 | 0.6277 | -1792.780 |
| bad | qrs_visibility | 0.2442 | 2 | 87.875 |
| bad | non_qrs_diff_p95 | 0.3754 | 1.473 | 55.345 |
| good | non_qrs_diff_p95 | 0.05031 | 1.473 | 36.190 |
| bad | flatline_ratio | 0.007206 | 0.09367 | 21.600 |
| medium | non_qrs_diff_p95 | 0.09235 | 1.473 | 20.243 |
| bad | baseline_step | 0.0282 | 0.1483 | 7.479 |
| bad | band_30_45 | 0.1049 | 0.01309 | -5.973 |
| medium | qrs_visibility | 0.2434 | 2 | 5.348 |
| good | qrs_visibility | 0.4924 | 2 | 3.848 |
| good | sqi_basSQI | 0.9466 | 0.6277 | -3.012 |
| bad | amplitude_entropy | 0.8964 | 0.7936 | -2.416 |
| good | detector_agreement | 0.2862 | 0.6667 | 2.095 |
| medium | detector_agreement | 0.2979 | 0.6667 | 2.051 |
| medium | sqi_basSQI | 0.9386 | 0.6277 | -1.646 |
| good | amplitude_entropy | 0.6282 | 0.7936 | 1.568 |
| bad | detector_agreement | 0.5147 | 0.6667 | 1.115 |
| good | flatline_ratio | 0.309 | 0.09367 | -1.031 |

## Interpretation Contract

- Good/medium/bad generation should be accepted only when waveform sheets look plausible and PCA/CDF gaps shrink together.
- If cross-stress disagrees with visual/distribution fit, cross-stress is a domain-shift note, not the optimization target.
- Atlas/KNN/PCA labels remain audit tools; the frozen model inference path remains waveform-derived channels only.