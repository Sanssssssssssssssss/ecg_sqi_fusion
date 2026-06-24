# v75fast_natural_source_nonbad Distribution-First Audit

- PTB protocol: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v75fast_natural_source_nonbad\protocol_v75fast_natural_source_nonbad_pc1500_s20260676`
- BUT reference: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\margin_ge_5s_keep_outlier_drop_mediumlike_bad_seed20260623`
- Model decision for the current line: freeze `Event-Factorized SQI Conformer / E1_query_only` as the simple waveform-only Transformer baseline.
- Cross-dataset scores are treated as stress diagnostics only; generation is judged first by visual waveform similarity and feature/PCA distribution overlap.

## Figures

- Shared PCA: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\distribution_first_audits\v75fast_natural_source_nonbad\v75fast_natural_source_nonbad_shared_pca_but_vs_ptb.png`
- Feature CDF overlap: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\distribution_first_audits\v75fast_natural_source_nonbad\v75fast_natural_source_nonbad_key_feature_cdf_overlap.png`
- Representative waveforms: `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\distribution_first_audits\v75fast_natural_source_nonbad\v75fast_natural_source_nonbad_representative_waveforms_by_label.png`

## Largest Median Gaps

| class | feature | BUT median | PTB median | PTB-BUT gap (BUT IQR) |
| --- | --- | ---: | ---: | ---: |
| bad | sqi_basSQI | 0.9997 | 0.8204 | -864.167 |
| bad | non_qrs_diff_p95 | 0.3754 | 1.979 | 80.842 |
| bad | qrs_visibility | 0.2442 | 1.029 | 39.257 |
| medium | non_qrs_diff_p95 | 0.09235 | 1.185 | 16.019 |
| good | non_qrs_diff_p95 | 0.05031 | 0.5433 | 12.539 |
| medium | qrs_visibility | 0.2434 | 1.469 | 3.733 |
| bad | band_30_45 | 0.1049 | 0.04903 | -3.634 |
| bad | detector_agreement | 0.5147 | 0.08333 | -3.165 |
| bad | amplitude_entropy | 0.8964 | 0.8019 | -2.221 |
| good | detector_agreement | 0.2862 | 0.6667 | 2.095 |
| bad | baseline_step | 0.0282 | 0.05474 | 1.653 |
| good | qrs_visibility | 0.4924 | 1.134 | 1.637 |
| medium | detector_agreement | 0.2979 | 0.5833 | 1.588 |
| good | amplitude_entropy | 0.6282 | 0.7869 | 1.504 |
| bad | flatline_ratio | 0.007206 | 0.01201 | 1.200 |
| good | flatline_ratio | 0.309 | 0.06405 | -1.172 |
| good | sqi_basSQI | 0.9466 | 0.8305 | -1.096 |
| good | band_30_45 | 0.01725 | 0.00501 | -1.059 |

## Interpretation Contract

- Good/medium/bad generation should be accepted only when waveform sheets look plausible and PCA/CDF gaps shrink together.
- If cross-stress disagrees with visual/distribution fit, cross-stress is a domain-shift note, not the optimization target.
- Atlas/KNN/PCA labels remain audit tools; the frozen model inference path remains waveform-derived channels only.