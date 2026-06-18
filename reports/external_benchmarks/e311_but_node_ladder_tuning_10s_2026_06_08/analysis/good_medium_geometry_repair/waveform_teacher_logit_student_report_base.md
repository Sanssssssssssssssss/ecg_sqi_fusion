# Waveform Teacher-Logit Student

The teacher uses synthetic 47-feature aux inputs.  Students use waveform-derived stats only and distill the teacher logits on synthetic data.  Original BUT is report-only.

## Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R |
|---|---|---:|---:|---:|---:|---:|
| feature_teacher_true_aux_reference | synthetic_test | 0.935930 | 0.900590 | 0.972803 | 0.961039 | 0.734440 |
| feature_teacher_true_aux_reference | original_test_all_10s+ | 0.910346 | 0.766230 | 0.872527 | 0.999096 | 0.289538 |
| feature_teacher_true_aux_reference | original_all_10s+ | 0.953332 | 0.953860 | 0.931585 | 0.998400 | 0.932829 |
| feature_teacher_true_aux_reference | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 |
| feature_teacher_true_aux_reference | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| logitstudent_balanced | synthetic_test | 0.992824 | 0.992156 | 0.979079 | 0.999188 | 0.987552 |
| logitstudent_balanced | original_test_all_10s+ | 0.816091 | 0.656034 | 0.853846 | 0.843877 | 0.182482 |
| logitstudent_balanced | original_all_10s+ | 0.806014 | 0.837537 | 0.708267 | 0.904403 | 0.923368 |
| logitstudent_balanced | bad_core_nearboundary | 0.630252 | 0.257732 | 0.000000 | 0.000000 | 0.630252 |
| logitstudent_balanced | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| logitstudent_medium_guard | synthetic_test | 0.994362 | 0.992926 | 0.991632 | 0.998377 | 0.979253 |
| logitstudent_medium_guard | original_test_all_10s+ | 0.813967 | 0.585518 | 0.871703 | 0.837551 | 0.048662 |
| logitstudent_medium_guard | original_all_10s+ | 0.805316 | 0.835784 | 0.711670 | 0.902051 | 0.912772 |
| logitstudent_medium_guard | bad_core_nearboundary | 0.168067 | 0.095923 | 0.000000 | 0.000000 | 0.168067 |
| logitstudent_medium_guard | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| logitstudent_bad_guard | synthetic_test | 0.991799 | 0.991190 | 0.979079 | 0.997565 | 0.987552 |
| logitstudent_bad_guard | original_test_all_10s+ | 0.830365 | 0.688344 | 0.825000 | 0.889968 | 0.236010 |
| logitstudent_bad_guard | original_all_10s+ | 0.818121 | 0.847554 | 0.720530 | 0.920211 | 0.927531 |
| logitstudent_bad_guard | bad_core_nearboundary | 0.815126 | 0.299383 | 0.000000 | 0.000000 | 0.815126 |
| logitstudent_bad_guard | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

## Files

- Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_teacher_logit_student_metrics_base.csv`
- Summary JSON: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_teacher_logit_student_summary_base.json`
