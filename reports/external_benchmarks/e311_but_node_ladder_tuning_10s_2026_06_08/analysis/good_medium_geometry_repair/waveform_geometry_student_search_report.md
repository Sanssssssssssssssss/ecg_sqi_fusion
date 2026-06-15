# Waveform Geometry Student

Input is waveform only. SQI/geometry features are teacher targets in the loss, not classifier inputs. Original BUT is report-only.

## Dirty Worktree Warning

Existing worktree changes were present before this runner. This experiment writes only external analysis/report/run outputs.

```text
M outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/boundary_blocks_build_summary.json
 M outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/boundary_blocks_diagnostic_promote_summary.json
 M outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/boundary_blocks_feature_summary.csv
 M outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/boundary_blocks_manifest.csv
 M outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/boundary_blocks_original_report_summary.json
 M outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/boundary_blocks_quick_summary.json
 M outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/boundary_blocks_report.md
 M outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/geometry_gate_last_run.json
 M outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/run_good_medium_geometry_repair.py
 M outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/run_waveform_geometry_student.py
 M outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/waveform_geometry_student_search_metrics.csv
 M outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/waveform_geometry_student_search_report.md
 M outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/waveform_geometry_student_smoke_metrics.csv
 M outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/waveform_geometry_student_smoke_report.md
 M outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/node_ladder_diagnostic_metrics.csv
 M outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/node_ladder_state.json
 M outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/node_promotion_decisions.csv
 M outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/node_registry.csv
 M outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/node_registry.json
 M reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/boundary_blocks_build_summary.json
 M reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/boundary_blocks_diagnostic_promote_summary.json
 M reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/boundary_blocks_feature_summary.csv
 M reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/boundary_blocks_manifest.csv
 M reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/boundary_blocks_original_report_summary.json
 M reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/boundary_blocks_quick_summary.json
 M reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/boundary_blocks_report.md
 M reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/waveform_geometry_student_search_report.md
 M reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/waveform_geometry_student_smoke_report.md
 M reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/node_ladder_diagn
```

## Metrics

| Candidate | Run | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | Teacher MAE | Core MAE |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| featurefusion_statfed_teacher | search | synthetic_val | 0.994764 | 0.993402 | 0.995086 | 0.999053 | 0.976562 | 2 | 1 | 6 | 0.7461 | 0.6532 |
| featurefusion_statfed_teacher | search | synthetic_test | 0.995387 | 0.993892 | 0.991632 | 1.000000 | 0.979253 | 4 | 0 | 5 | 0.7408 | 0.6123 |
| featurefusion_statfed_teacher_badcal | search | synthetic_test | 0.995387 | 0.993337 | 0.991632 | 1.000000 | 0.979253 | 3 | 0 | 5 | nan | nan |
| featurefusion_statfed_teacher | search | original_test_all_10s+ | 0.772915 | 0.544508 | 0.854121 | 0.775418 | 0.026764 | 528 | 940 | 195 | nan | nan |
| featurefusion_statfed_teacher_badcal | search | original_test_all_10s+ | 0.772443 | 0.564406 | 0.851099 | 0.772933 | 0.070560 | 528 | 892 | 186 | nan | nan |
| featurefusion_statfed_teacher | search | original_all_10s+ | 0.773091 | 0.808685 | 0.658393 | 0.888690 | 0.910501 | 5811 | 1106 | 266 | nan | nan |
| featurefusion_statfed_teacher_badcal | search | original_all_10s+ | 0.771210 | 0.804464 | 0.655166 | 0.885491 | 0.915610 | 5805 | 1050 | 248 | nan | nan |
| featurefusion_statfed_teacher | search | bad_core_nearboundary | 0.008403 | 0.005556 | 0.000000 | 0.000000 | 0.008403 | 0 | 0 | 118 | nan | nan |
| featurefusion_statfed_teacher_badcal | search | bad_core_nearboundary | 0.084034 | 0.051680 | 0.000000 | 0.000000 | 0.084034 | 0 | 0 | 109 | nan | nan |
| featurefusion_statfed_teacher | search | bad_outlier_stress | 0.034247 | 0.022075 | 0.000000 | 0.000000 | 0.034247 | 0 | 0 | 77 | nan | nan |
| featurefusion_statfed_teacher_badcal | search | bad_outlier_stress | 0.065068 | 0.040729 | 0.000000 | 0.000000 | 0.065068 | 0 | 0 | 77 | nan | nan |
| featurefusion_stressbank_teacher | search | synthetic_val | 0.994183 | 0.992836 | 0.995086 | 0.998106 | 0.976562 | 2 | 2 | 6 | 0.7475 | 0.6495 |
| featurefusion_stressbank_teacher | search | synthetic_test | 0.995387 | 0.993892 | 0.991632 | 1.000000 | 0.979253 | 4 | 0 | 5 | 0.7362 | 0.6159 |
| featurefusion_stressbank_teacher_badcal | search | synthetic_test | 0.986674 | 0.975946 | 0.956067 | 1.000000 | 0.979253 | 3 | 0 | 5 | nan | nan |
| featurefusion_stressbank_teacher | search | original_test_all_10s+ | 0.802289 | 0.689333 | 0.900000 | 0.767058 | 0.316302 | 361 | 980 | 58 | nan | nan |
| featurefusion_stressbank_teacher_badcal | search | original_test_all_10s+ | 0.800401 | 0.678255 | 0.895604 | 0.764347 | 0.345499 | 359 | 905 | 58 | nan | nan |
| featurefusion_stressbank_teacher | search | original_all_10s+ | 0.828529 | 0.853457 | 0.766590 | 0.875047 | 0.934721 | 3969 | 1257 | 120 | nan | nan |
| featurefusion_stressbank_teacher_badcal | search | original_all_10s+ | 0.825495 | 0.846762 | 0.761544 | 0.872318 | 0.937559 | 3953 | 1166 | 118 | nan | nan |
| featurefusion_stressbank_teacher | search | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | nan | nan |
| featurefusion_stressbank_teacher_badcal | search | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | nan | nan |
| featurefusion_stressbank_teacher | search | bad_outlier_stress | 0.037671 | 0.024202 | 0.000000 | 0.000000 | 0.037671 | 0 | 0 | 58 | nan | nan |
| featurefusion_stressbank_teacher_badcal | search | bad_outlier_stress | 0.078767 | 0.048677 | 0.000000 | 0.000000 | 0.078767 | 0 | 0 | 58 | nan | nan |
| featurefusion_multiscale_core | search | synthetic_val | 0.993601 | 0.992260 | 0.987715 | 1.000000 | 0.976562 | 5 | 0 | 6 | 0.7449 | 0.6422 |
| featurefusion_multiscale_core | search | synthetic_test | 0.993849 | 0.992432 | 0.985356 | 1.000000 | 0.979253 | 7 | 0 | 5 | 0.7388 | 0.6188 |
| featurefusion_multiscale_core_badcal | search | synthetic_test | 0.993849 | 0.992432 | 0.985356 | 1.000000 | 0.979253 | 7 | 0 | 5 | nan | nan |
| featurefusion_multiscale_core | search | original_test_all_10s+ | 0.812434 | 0.681101 | 0.874176 | 0.812020 | 0.270073 | 458 | 785 | 103 | nan | nan |
| featurefusion_multiscale_core_badcal | search | original_test_all_10s+ | 0.807479 | 0.676055 | 0.868956 | 0.799593 | 0.347932 | 452 | 698 | 84 | nan | nan |
| featurefusion_multiscale_core | search | original_all_10s+ | 0.798064 | 0.830860 | 0.691310 | 0.902992 | 0.931315 | 5257 | 973 | 164 | nan | nan |
| featurefusion_multiscale_core_badcal | search | original_all_10s+ | 0.793148 | 0.820715 | 0.684915 | 0.894900 | 0.937559 | 5230 | 868 | 145 | nan | nan |
| featurefusion_multiscale_core | search | bad_core_nearboundary | 0.857143 | 0.307692 | 0.000000 | 0.000000 | 0.857143 | 0 | 0 | 17 | nan | nan |
| featurefusion_multiscale_core_badcal | search | bad_core_nearboundary | 0.983193 | 0.330508 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 2 | nan | nan |
| featurefusion_multiscale_core | search | bad_outlier_stress | 0.030822 | 0.019934 | 0.000000 | 0.000000 | 0.030822 | 0 | 0 | 86 | nan | nan |
| featurefusion_multiscale_core_badcal | search | bad_outlier_stress | 0.089041 | 0.054507 | 0.000000 | 0.000000 | 0.089041 | 0 | 0 | 82 | nan | nan |

## Baselines

- `47_feature_tabular_upper_bound` original_test_all_10s+: acc `0.963548`, recalls 0.956/0.973/0.927`. tabular-only upper bound; geometry features are inference inputs.
- `current_best_waveform_transformer` original_test_all_10s+: acc `0.811018`, recalls 0.961/0.726/0.401`. aug_convtx_balanced_focal raw.
- `stattoken_baseline` original_test_all_10s+: acc `0.813141`, recalls 0.962/0.726/0.433`. stattx_medium_guard raw.
- `reduced_feature_auto_topk20` original_test_all_10s+: acc `0.913531`, recalls 0.973/0.885/0.696`. reduced tabular model; not waveform-only.

## Candidates

- `featurefusion_statfed_teacher` (search): best_epoch=8, useful_gate=pass, original_report_ran=True, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_geometry_student\N17043_gm_probe\search\featurefusion_statfed_teacher\ckpt_best.pt`
- `featurefusion_stressbank_teacher` (search): best_epoch=8, useful_gate=pass, original_report_ran=True, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_geometry_student\N17043_gm_probe\search\featurefusion_stressbank_teacher\ckpt_best.pt`
- `featurefusion_multiscale_core` (search): best_epoch=7, useful_gate=pass, original_report_ran=True, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_geometry_student\N17043_gm_probe\search\featurefusion_multiscale_core\ckpt_best.pt`

## Notes

- Selection uses synthetic/node diagnostic only.
- BUT buckets are emitted only for useful-gate candidates unless `--but-report-mode always` is used.
- Diagnostic artifacts are saved beside each checkpoint: teacher recovery CSV, error rows, embedding PCA, and waveform mistake panels.
- Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_geometry_student_search_metrics.csv`
