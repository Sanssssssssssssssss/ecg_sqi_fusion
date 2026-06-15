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
| neighbor_stattoken_student | search | synthetic_val | 0.995346 | 0.993970 | 1.000000 | 0.998106 | 0.976562 | 0 | 2 | 6 | 0.7698 | 0.6785 |
| neighbor_stattoken_student | search | synthetic_test | 0.993337 | 0.991953 | 0.987448 | 0.998377 | 0.979253 | 6 | 2 | 5 | 0.7567 | 0.6323 |
| neighbor_stattoken_student_badcal | search | synthetic_test | 0.993337 | 0.991953 | 0.987448 | 0.998377 | 0.979253 | 6 | 2 | 5 | nan | nan |
| neighbor_stattoken_student | search | original_test_all_10s+ | 0.765129 | 0.521563 | 0.761813 | 0.838906 | 0.000000 | 867 | 713 | 318 | nan | nan |
| neighbor_stattoken_student_badcal | search | original_test_all_10s+ | 0.777870 | 0.663403 | 0.761813 | 0.838906 | 0.262774 | 867 | 713 | 210 | nan | nan |
| neighbor_stattoken_student | search | original_all_10s+ | 0.438828 | 0.308512 | 0.268497 | 0.930184 | 0.000000 | 12467 | 742 | 5190 | nan | nan |
| neighbor_stattoken_student_badcal | search | original_all_10s+ | 0.466410 | 0.410561 | 0.268497 | 0.930184 | 0.171996 | 12467 | 742 | 4281 | nan | nan |
| neighbor_stattoken_student | search | bad_core_nearboundary | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 119 | nan | nan |
| neighbor_stattoken_student_badcal | search | bad_core_nearboundary | 0.907563 | 0.317181 | 0.000000 | 0.000000 | 0.907563 | 0 | 0 | 11 | nan | nan |
| neighbor_stattoken_student | search | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 199 | nan | nan |
| neighbor_stattoken_student_badcal | search | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 199 | nan | nan |
| neighbor_hybrid_atlas_student | search | synthetic_val | 0.993601 | 0.993057 | 1.000000 | 0.991477 | 0.992188 | 0 | 8 | 2 | 0.7658 | 0.6532 |
| neighbor_hybrid_atlas_student | search | synthetic_test | 0.996412 | 0.995574 | 0.997908 | 0.997565 | 0.987552 | 1 | 3 | 3 | 0.7617 | 0.6226 |
| neighbor_hybrid_atlas_student_badcal | search | synthetic_test | 0.996925 | 0.996413 | 0.997908 | 0.996753 | 0.995851 | 1 | 3 | 1 | nan | nan |
| neighbor_hybrid_atlas_student | search | original_test_all_10s+ | 0.794267 | 0.672854 | 0.912088 | 0.747402 | 0.255474 | 320 | 1116 | 88 | nan | nan |
| neighbor_hybrid_atlas_student_badcal | search | original_test_all_10s+ | 0.793913 | 0.681862 | 0.912088 | 0.743561 | 0.289538 | 320 | 1116 | 74 | nan | nan |
| neighbor_hybrid_atlas_student | search | original_all_10s+ | 0.680908 | 0.501652 | 0.773807 | 0.860651 | 0.019868 | 3855 | 1474 | 4956 | nan | nan |
| neighbor_hybrid_atlas_student_badcal | search | original_all_10s+ | 0.680453 | 0.503127 | 0.773807 | 0.857734 | 0.022895 | 3855 | 1474 | 4940 | nan | nan |
| neighbor_hybrid_atlas_student | search | bad_core_nearboundary | 0.882353 | 0.312500 | 0.000000 | 0.000000 | 0.882353 | 0 | 0 | 14 | nan | nan |
| neighbor_hybrid_atlas_student_badcal | search | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | nan | nan |
| neighbor_hybrid_atlas_student | search | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 74 | nan | nan |
| neighbor_hybrid_atlas_student_badcal | search | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 74 | nan | nan |

## Baselines

- `47_feature_tabular_upper_bound` original_test_all_10s+: acc `0.963548`, recalls 0.956/0.973/0.927`. tabular-only upper bound; geometry features are inference inputs.
- `current_best_waveform_transformer` original_test_all_10s+: acc `0.811018`, recalls 0.961/0.726/0.401`. aug_convtx_balanced_focal raw.
- `stattoken_baseline` original_test_all_10s+: acc `0.813141`, recalls 0.962/0.726/0.433`. stattx_medium_guard raw.
- `reduced_feature_auto_topk20` original_test_all_10s+: acc `0.913531`, recalls 0.973/0.885/0.696`. reduced tabular model; not waveform-only.

## Candidates

- `neighbor_stattoken_student` (search): best_epoch=8, useful_gate=pass, original_report_ran=True, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_geometry_student\N17043_gm_probe\search\neighbor_stattoken_student\ckpt_best.pt`
- `neighbor_hybrid_atlas_student` (search): best_epoch=8, useful_gate=pass, original_report_ran=True, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_geometry_student\N17043_gm_probe\search\neighbor_hybrid_atlas_student\ckpt_best.pt`

## Notes

- Selection uses synthetic/node diagnostic only.
- BUT buckets are emitted only for useful-gate candidates unless `--but-report-mode always` is used.
- Diagnostic artifacts are saved beside each checkpoint: teacher recovery CSV, error rows, embedding PCA, and waveform mistake panels.
- Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_geometry_student_search_metrics.csv`
