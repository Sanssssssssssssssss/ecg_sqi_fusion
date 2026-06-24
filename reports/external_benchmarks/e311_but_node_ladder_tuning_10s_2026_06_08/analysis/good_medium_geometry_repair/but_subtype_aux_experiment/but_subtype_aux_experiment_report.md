# BUT Subtype-Auxiliary Waveform Experiment

- Generated: 2026-06-21 00:14:55
- Protocol: `outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\clean_but_protocols\margin_ge_5s_keep_outlier_subtype_stratified_seed20260621`
- Inference input: waveform-derived channels only.
- Formal output remains 3-class good/medium/bad.
- Subtype labels are auxiliary supervision and diagnostics only.

## Split Counts

| split | class_name | n |
| --- | --- | --- |
| test | bad | 1031 |
| test | good | 3008 |
| test | medium | 1842 |
| train | bad | 3608 |
| train | good | 10530 |
| train | medium | 6449 |
| val | bad | 517 |
| val | good | 1504 |
| val | medium | 921 |

## Subtype Counts

| split | class_name | subtype | n |
| --- | --- | --- | --- |
| test | bad | bad_baseline_wander_lowfreq | 15 |
| test | bad | bad_contact_reset_flatline | 47 |
| test | bad | bad_dense_right_island | 631 |
| test | bad | bad_detector_template_disagree | 229 |
| test | bad | bad_highfreq_detail_noise | 14 |
| test | bad | bad_low_qrs_visibility | 10 |
| test | bad | bad_other_boundary | 85 |
| test | good | good_overlap_or_mild_artifact | 3008 |
| test | medium | medium_overlap_or_detail | 1842 |
| train | bad | bad_baseline_wander_lowfreq | 54 |
| train | bad | bad_contact_reset_flatline | 164 |
| train | bad | bad_dense_right_island | 2208 |
| train | bad | bad_detector_template_disagree | 803 |
| train | bad | bad_highfreq_detail_noise | 48 |
| train | bad | bad_low_qrs_visibility | 33 |
| train | bad | bad_other_boundary | 298 |
| train | good | good_overlap_or_mild_artifact | 10530 |
| train | medium | medium_overlap_or_detail | 6448 |
| train | medium | medium_stable | 1 |
| val | bad | bad_baseline_wander_lowfreq | 8 |
| val | bad | bad_contact_reset_flatline | 24 |
| val | bad | bad_dense_right_island | 316 |
| val | bad | bad_detector_template_disagree | 115 |
| val | bad | bad_highfreq_detail_noise | 7 |
| val | bad | bad_low_qrs_visibility | 5 |
| val | bad | bad_other_boundary | 42 |
| val | good | good_overlap_or_mild_artifact | 1504 |
| val | medium | medium_overlap_or_detail | 921 |

## Results

| candidate | bucket | runs | acc | acc_std | macro_f1 | macro_f1_std | good_recall | good_recall_std | medium_recall | medium_recall_std | bad_recall | bad_recall_std | subtype_acc | subtype_acc_std | bad_subtype_acc | bad_subtype_acc_std | good_to_medium | good_to_medium_std | medium_to_good | medium_to_good_std | bad_to_nonbad | bad_to_nonbad_std | nonbad_to_bad | nonbad_to_bad_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1_baseline_keep | test | 1 | 0.927903 | 0 | 0.935338 | 0 | 0.926862 | 0 | 0.900109 | 0 | 0.980601 | 0 | 0.0394491 | 0 | 0 | 0 | 220 | 0 | 174 | 0 | 20 | 0 | 10 | 0 |
| E1_subtype_aux_keep | test | 1 | 0.926713 | 0 | 0.93349 | 0 | 0.910239 | 0 | 0.913681 | 0 | 0.99806 | 0 | 0.820099 | 0 | 0.414161 | 0 | 270 | 0 | 120 | 0 | 2 | 0 | 39 | 0 |
| E4_local_art_subtype_keep | test | 1 | 0.921442 | 0 | 0.930114 | 0 | 0.912899 | 0 | 0.897937 | 0 | 0.988361 | 0 | 0.823839 | 0 | 0.43453 | 0 | 262 | 0 | 171 | 0 | 12 | 0 | 17 | 0 |
| E1_baseline_keep | val | 1 | 0.924541 | 0 | 0.933805 | 0 | 0.919548 | 0 | 0.897937 | 0 | 0.98646 | 0 | 0.0329708 | 0 | 0 | 0 | 121 | 0 | 92 | 0 | 7 | 0 | 2 | 0 |
| E1_subtype_aux_keep | val | 1 | 0.922502 | 0 | 0.93094 | 0 | 0.899601 | 0 | 0.918567 | 0 | 0.996132 | 0 | 0.811693 | 0 | 0.415861 | 0 | 150 | 0 | 61 | 0 | 2 | 0 | 15 | 0 |
| E4_local_art_subtype_keep | val | 1 | 0.919782 | 0 | 0.928546 | 0 | 0.90758 | 0 | 0.90228 | 0 | 0.98646 | 0 | 0.823249 | 0 | 0.464217 | 0 | 139 | 0 | 81 | 0 | 7 | 0 | 9 | 0 |

## Output Files

- Metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_subtype_aux_experiment\but_subtype_aux_metrics.csv`
- Subtype metrics: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_subtype_aux_experiment\but_subtype_aux_subtype_metrics.csv`
- Bad mechanism audit: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\but_subtype_aux_experiment\but_bad_subtype_multilabel_audit.csv`
- Checkpoints/logs: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\but_subtype_aux_experiment`