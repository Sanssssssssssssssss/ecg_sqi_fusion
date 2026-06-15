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
| stresstoken_statfed_atlas | search | synthetic_val | 0.995346 | 0.993970 | 1.000000 | 0.998106 | 0.976562 | 0 | 2 | 6 | 0.7513 | 0.6628 |
| stresstoken_statfed_atlas | search | synthetic_test | 0.992824 | 0.991485 | 0.993724 | 0.995130 | 0.979253 | 3 | 6 | 5 | 0.7484 | 0.6287 |
| stresstoken_statfed_atlas_badcal | search | synthetic_test | 0.992824 | 0.991485 | 0.993724 | 0.995130 | 0.979253 | 3 | 6 | 5 | nan | nan |
| stresstoken_statfed_atlas | search | original_test_all_10s+ | 0.749322 | 0.533105 | 0.915934 | 0.678717 | 0.034063 | 306 | 1422 | 139 | nan | nan |
| stresstoken_statfed_atlas_badcal | search | original_test_all_10s+ | 0.753804 | 0.587251 | 0.915934 | 0.678717 | 0.126521 | 306 | 1422 | 101 | nan | nan |
| stresstoken_statfed_atlas | search | original_all_10s+ | 0.769511 | 0.751159 | 0.802265 | 0.821509 | 0.559319 | 3370 | 1897 | 2053 | nan | nan |
| stresstoken_statfed_atlas_badcal | search | original_all_10s+ | 0.826344 | 0.849529 | 0.802265 | 0.821321 | 0.914096 | 3370 | 1897 | 178 | nan | nan |
| stresstoken_statfed_atlas | search | bad_core_nearboundary | 0.117647 | 0.070175 | 0.000000 | 0.000000 | 0.117647 | 0 | 0 | 105 | nan | nan |
| stresstoken_statfed_atlas_badcal | search | bad_core_nearboundary | 0.436975 | 0.202729 | 0.000000 | 0.000000 | 0.436975 | 0 | 0 | 67 | nan | nan |
| stresstoken_statfed_atlas | search | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 34 | nan | nan |
| stresstoken_statfed_atlas_badcal | search | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 34 | nan | nan |
| stresstoken_multiscale_atlas | search | synthetic_val | 0.994183 | 0.992836 | 0.995086 | 0.998106 | 0.976562 | 2 | 2 | 6 | 0.7656 | 0.6580 |
| stresstoken_multiscale_atlas | search | synthetic_test | 0.994362 | 0.992926 | 0.991632 | 0.998377 | 0.979253 | 4 | 2 | 5 | 0.7520 | 0.6200 |
| stresstoken_multiscale_atlas_badcal | search | synthetic_test | 0.994362 | 0.992926 | 0.991632 | 0.998377 | 0.979253 | 4 | 2 | 5 | nan | nan |
| stresstoken_multiscale_atlas | search | original_test_all_10s+ | 0.787543 | 0.672866 | 0.902473 | 0.741301 | 0.267640 | 355 | 1142 | 56 | nan | nan |
| stresstoken_multiscale_atlas_badcal | search | original_test_all_10s+ | 0.788604 | 0.681993 | 0.902473 | 0.741075 | 0.291971 | 355 | 1139 | 47 | nan | nan |
| stresstoken_multiscale_atlas | search | original_all_10s+ | 0.814328 | 0.843668 | 0.746465 | 0.865450 | 0.930369 | 4321 | 1427 | 119 | nan | nan |
| stresstoken_multiscale_atlas_badcal | search | original_all_10s+ | 0.814692 | 0.844116 | 0.746465 | 0.865262 | 0.933018 | 4321 | 1424 | 108 | nan | nan |
| stresstoken_multiscale_atlas | search | bad_core_nearboundary | 0.924370 | 0.320233 | 0.000000 | 0.000000 | 0.924370 | 0 | 0 | 9 | nan | nan |
| stresstoken_multiscale_atlas_badcal | search | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | nan | nan |
| stresstoken_multiscale_atlas | search | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 47 | nan | nan |
| stresstoken_multiscale_atlas_badcal | search | bad_outlier_stress | 0.003425 | 0.002275 | 0.000000 | 0.000000 | 0.003425 | 0 | 0 | 47 | nan | nan |

## Baselines

- `47_feature_tabular_upper_bound` original_test_all_10s+: acc `0.963548`, recalls 0.956/0.973/0.927`. tabular-only upper bound; geometry features are inference inputs.
- `current_best_waveform_transformer` original_test_all_10s+: acc `0.811018`, recalls 0.961/0.726/0.401`. aug_convtx_balanced_focal raw.
- `stattoken_baseline` original_test_all_10s+: acc `0.813141`, recalls 0.962/0.726/0.433`. stattx_medium_guard raw.
- `reduced_feature_auto_topk20` original_test_all_10s+: acc `0.913531`, recalls 0.973/0.885/0.696`. reduced tabular model; not waveform-only.

## Candidates

- `stresstoken_statfed_atlas` (search): best_epoch=6, useful_gate=pass, original_report_ran=True, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_geometry_student\N17043_gm_probe\search\stresstoken_statfed_atlas\ckpt_best.pt`
- `stresstoken_multiscale_atlas` (search): best_epoch=8, useful_gate=pass, original_report_ran=True, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_geometry_student\N17043_gm_probe\search\stresstoken_multiscale_atlas\ckpt_best.pt`

## Notes

- Selection uses synthetic/node diagnostic only.
- BUT buckets are emitted only for useful-gate candidates unless `--but-report-mode always` is used.
- Diagnostic artifacts are saved beside each checkpoint: teacher recovery CSV, error rows, embedding PCA, and waveform mistake panels.
- Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_geometry_student_search_metrics.csv`
