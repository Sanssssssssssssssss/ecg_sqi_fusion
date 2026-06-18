# Waveform Teacher-Logit Student

The teacher uses synthetic 47-feature aux inputs.  Students use waveform-derived stats only and distill the teacher logits on synthetic data.  Original BUT is report-only.

## Metrics

| Candidate | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R |
|---|---|---:|---:|---:|---:|---:|
| feature_teacher_true_aux_reference | synthetic_test | 0.929267 | 0.894382 | 0.945607 | 0.961039 | 0.734440 |
| feature_teacher_true_aux_reference | original_test_all_10s+ | 0.913413 | 0.769021 | 0.879396 | 0.999322 | 0.289538 |
| feature_teacher_true_aux_reference | original_all_10s+ | 0.940011 | 0.943340 | 0.905709 | 0.998400 | 0.933207 |
| feature_teacher_true_aux_reference | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 |
| feature_teacher_true_aux_reference | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| logitstudent_balanced | synthetic_test | 0.986674 | 0.982862 | 0.972803 | 0.993506 | 0.979253 |
| logitstudent_balanced | original_test_all_10s+ | 0.708623 | 0.563928 | 0.583791 | 0.841844 | 0.379562 |
| logitstudent_balanced | original_all_10s+ | 0.752701 | 0.767536 | 0.603356 | 0.899699 | 0.938694 |
| logitstudent_balanced | bad_core_nearboundary | 0.546218 | 0.235507 | 0.000000 | 0.000000 | 0.546218 |
| logitstudent_balanced | bad_outlier_stress | 0.311644 | 0.158399 | 0.000000 | 0.000000 | 0.311644 |
| logitstudent_medium_guard | synthetic_test | 0.990261 | 0.985645 | 0.987448 | 0.993506 | 0.979253 |
| logitstudent_medium_guard | original_test_all_10s+ | 0.712162 | 0.591915 | 0.689011 | 0.746498 | 0.547445 |
| logitstudent_medium_guard | original_all_10s+ | 0.746116 | 0.752489 | 0.614798 | 0.854065 | 0.952507 |
| logitstudent_medium_guard | bad_core_nearboundary | 0.369748 | 0.179959 | 0.000000 | 0.000000 | 0.369748 |
| logitstudent_medium_guard | bad_outlier_stress | 0.619863 | 0.255109 | 0.000000 | 0.000000 | 0.619863 |
| logitstudent_bad_guard | synthetic_test | 0.981035 | 0.977120 | 0.953975 | 0.991883 | 0.979253 |
| logitstudent_bad_guard | original_test_all_10s+ | 0.746726 | 0.582138 | 0.640110 | 0.877316 | 0.284672 |
| logitstudent_bad_guard | original_all_10s+ | 0.752215 | 0.771549 | 0.593323 | 0.918799 | 0.929612 |
| logitstudent_bad_guard | bad_core_nearboundary | 0.201681 | 0.111888 | 0.000000 | 0.000000 | 0.201681 |
| logitstudent_bad_guard | bad_outlier_stress | 0.318493 | 0.161039 | 0.000000 | 0.000000 | 0.318493 |

## Files

- Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_teacher_logit_student_metrics_aug.csv`
- Summary JSON: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_teacher_logit_student_summary_aug.json`
