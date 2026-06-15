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
| multiscale_statpatch_stress_shell | search | synthetic_val | 0.995346 | 0.993969 | 0.995086 | 1.000000 | 0.976562 | 2 | 0 | 6 | 0.7677 | 0.6458 |
| multiscale_statpatch_stress_shell | search | synthetic_test | 0.995387 | 0.993892 | 0.991632 | 1.000000 | 0.979253 | 4 | 0 | 5 | 0.7592 | 0.6158 |
| multiscale_statpatch_stress_shell_badcal | search | synthetic_test | 0.995387 | 0.993892 | 0.991632 | 1.000000 | 0.979253 | 4 | 0 | 5 | nan | nan |
| multiscale_statpatch_stress_shell | search | original_test_all_10s+ | 0.765483 | 0.604574 | 0.889011 | 0.721419 | 0.145985 | 404 | 1230 | 112 | nan | nan |
| multiscale_statpatch_stress_shell_badcal | search | original_test_all_10s+ | 0.768196 | 0.659436 | 0.887637 | 0.713285 | 0.301703 | 400 | 1217 | 55 | nan | nan |
| multiscale_statpatch_stress_shell | search | original_all_10s+ | 0.777370 | 0.814288 | 0.679458 | 0.863568 | 0.919773 | 5462 | 1443 | 183 | nan | nan |
| multiscale_statpatch_stress_shell_badcal | search | original_all_10s+ | 0.777855 | 0.814251 | 0.678930 | 0.858675 | 0.934342 | 5458 | 1426 | 114 | nan | nan |
| multiscale_statpatch_stress_shell | search | bad_core_nearboundary | 0.504202 | 0.223464 | 0.000000 | 0.000000 | 0.504202 | 0 | 0 | 59 | nan | nan |
| multiscale_statpatch_stress_shell_badcal | search | bad_core_nearboundary | 0.941176 | 0.323232 | 0.000000 | 0.000000 | 0.941176 | 0 | 0 | 7 | nan | nan |
| multiscale_statpatch_stress_shell | search | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 53 | nan | nan |
| multiscale_statpatch_stress_shell_badcal | search | bad_outlier_stress | 0.041096 | 0.026316 | 0.000000 | 0.000000 | 0.041096 | 0 | 0 | 48 | nan | nan |
| statfed_patch_stress_shell | search | synthetic_val | 0.993601 | 0.992264 | 0.990172 | 0.999053 | 0.976562 | 4 | 1 | 6 | 0.7521 | 0.6470 |
| statfed_patch_stress_shell | search | synthetic_test | 0.994874 | 0.993406 | 0.989540 | 1.000000 | 0.979253 | 5 | 0 | 5 | 0.7453 | 0.6158 |
| statfed_patch_stress_shell_badcal | search | synthetic_test | 0.994362 | 0.991478 | 0.989540 | 0.999188 | 0.979253 | 3 | 0 | 5 | nan | nan |
| statfed_patch_stress_shell | search | original_test_all_10s+ | 0.800047 | 0.649800 | 0.856868 | 0.809761 | 0.192214 | 521 | 842 | 114 | nan | nan |
| statfed_patch_stress_shell_badcal | search | original_test_all_10s+ | 0.803940 | 0.692677 | 0.856868 | 0.807049 | 0.301703 | 518 | 838 | 73 | nan | nan |
| statfed_patch_stress_shell | search | original_all_10s+ | 0.783924 | 0.820610 | 0.664848 | 0.905438 | 0.923557 | 5711 | 1004 | 183 | nan | nan |
| statfed_patch_stress_shell_badcal | search | original_all_10s+ | 0.783560 | 0.817989 | 0.664672 | 0.899323 | 0.934153 | 5632 | 998 | 133 | nan | nan |
| statfed_patch_stress_shell | search | bad_core_nearboundary | 0.663866 | 0.265993 | 0.000000 | 0.000000 | 0.663866 | 0 | 0 | 40 | nan | nan |
| statfed_patch_stress_shell_badcal | search | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | nan | nan |
| statfed_patch_stress_shell | search | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 74 | nan | nan |
| statfed_patch_stress_shell_badcal | search | bad_outlier_stress | 0.017123 | 0.011223 | 0.000000 | 0.000000 | 0.017123 | 0 | 0 | 73 | nan | nan |

## Baselines

- `47_feature_tabular_upper_bound` original_test_all_10s+: acc `0.963548`, recalls 0.956/0.973/0.927`. tabular-only upper bound; geometry features are inference inputs.
- `current_best_waveform_transformer` original_test_all_10s+: acc `0.811018`, recalls 0.961/0.726/0.401`. aug_convtx_balanced_focal raw.
- `stattoken_baseline` original_test_all_10s+: acc `0.813141`, recalls 0.962/0.726/0.433`. stattx_medium_guard raw.
- `reduced_feature_auto_topk20` original_test_all_10s+: acc `0.913531`, recalls 0.973/0.885/0.696`. reduced tabular model; not waveform-only.

## Candidates

- `multiscale_statpatch_stress_shell` (search): best_epoch=7, useful_gate=pass, original_report_ran=True, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_geometry_student\N17043_gm_probe\search\multiscale_statpatch_stress_shell\ckpt_best.pt`
- `statfed_patch_stress_shell` (search): best_epoch=7, useful_gate=pass, original_report_ran=True, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_geometry_student\N17043_gm_probe\search\statfed_patch_stress_shell\ckpt_best.pt`

## Notes

- Selection uses synthetic/node diagnostic only.
- BUT buckets are emitted only for useful-gate candidates unless `--but-report-mode always` is used.
- Diagnostic artifacts are saved beside each checkpoint: teacher recovery CSV, error rows, embedding PCA, and waveform mistake panels.
- Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_geometry_student_search_metrics.csv`
