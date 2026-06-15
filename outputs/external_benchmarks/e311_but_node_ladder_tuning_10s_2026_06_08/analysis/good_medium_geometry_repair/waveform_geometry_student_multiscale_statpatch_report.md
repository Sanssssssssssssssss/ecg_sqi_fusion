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
| multiscale_statpatch_balanced | search | synthetic_val | 0.992437 | 0.991402 | 0.995086 | 0.994318 | 0.980469 | 2 | 6 | 5 | 0.7718 | 0.6499 |
| multiscale_statpatch_balanced | search | synthetic_test | 0.994362 | 0.992929 | 0.993724 | 0.997565 | 0.979253 | 3 | 3 | 5 | 0.7653 | 0.6198 |
| multiscale_statpatch_balanced_badcal | search | synthetic_test | 0.994362 | 0.992929 | 0.993724 | 0.997565 | 0.979253 | 3 | 3 | 5 | nan | nan |
| multiscale_statpatch_balanced | search | original_test_all_10s+ | 0.767960 | 0.667084 | 0.915934 | 0.690917 | 0.287105 | 306 | 1366 | 71 | nan | nan |
| multiscale_statpatch_balanced_badcal | search | original_test_all_10s+ | 0.767960 | 0.667084 | 0.915934 | 0.690917 | 0.287105 | 306 | 1366 | 71 | nan | nan |
| multiscale_statpatch_balanced | search | original_all_10s+ | 0.840848 | 0.862718 | 0.819281 | 0.830166 | 0.931883 | 3080 | 1799 | 137 | nan | nan |
| multiscale_statpatch_balanced_badcal | search | original_all_10s+ | 0.840879 | 0.862763 | 0.819281 | 0.830166 | 0.932072 | 3080 | 1799 | 136 | nan | nan |
| multiscale_statpatch_balanced | search | bad_core_nearboundary | 0.991597 | 0.331927 | 0.000000 | 0.000000 | 0.991597 | 0 | 0 | 1 | nan | nan |
| multiscale_statpatch_balanced_badcal | search | bad_core_nearboundary | 0.991597 | 0.331927 | 0.000000 | 0.000000 | 0.991597 | 0 | 0 | 1 | nan | nan |
| multiscale_statpatch_balanced | search | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 70 | nan | nan |
| multiscale_statpatch_balanced_badcal | search | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 70 | nan | nan |
| multiscale_statpatch_medium_badguard | search | synthetic_val | 0.994183 | 0.992836 | 0.995086 | 0.998106 | 0.976562 | 2 | 2 | 6 | 0.7387 | 0.6405 |
| multiscale_statpatch_medium_badguard | search | synthetic_test | 0.994362 | 0.992929 | 0.993724 | 0.997565 | 0.979253 | 3 | 3 | 5 | 0.7425 | 0.6174 |
| multiscale_statpatch_medium_badguard_badcal | search | synthetic_test | 0.994362 | 0.992929 | 0.993724 | 0.997565 | 0.979253 | 3 | 3 | 5 | nan | nan |
| multiscale_statpatch_medium_badguard | search | original_test_all_10s+ | 0.805002 | 0.693970 | 0.903571 | 0.771803 | 0.289538 | 351 | 1010 | 66 | nan | nan |
| multiscale_statpatch_medium_badguard_badcal | search | original_test_all_10s+ | 0.804884 | 0.693640 | 0.903571 | 0.771577 | 0.289538 | 351 | 1010 | 66 | nan | nan |
| multiscale_statpatch_medium_badguard | search | original_all_10s+ | 0.835569 | 0.860309 | 0.782315 | 0.872695 | 0.932640 | 3710 | 1353 | 128 | nan | nan |
| multiscale_statpatch_medium_badguard_badcal | search | original_all_10s+ | 0.835144 | 0.859587 | 0.782315 | 0.870342 | 0.934721 | 3710 | 1353 | 117 | nan | nan |
| multiscale_statpatch_medium_badguard | search | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | nan | nan |
| multiscale_statpatch_medium_badguard_badcal | search | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | nan | nan |
| multiscale_statpatch_medium_badguard | search | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 66 | nan | nan |
| multiscale_statpatch_medium_badguard_badcal | search | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 66 | nan | nan |

## Baselines

- `47_feature_tabular_upper_bound` original_test_all_10s+: acc `0.963548`, recalls 0.956/0.973/0.927`. tabular-only upper bound; geometry features are inference inputs.
- `current_best_waveform_transformer` original_test_all_10s+: acc `0.811018`, recalls 0.961/0.726/0.401`. aug_convtx_balanced_focal raw.
- `stattoken_baseline` original_test_all_10s+: acc `0.813141`, recalls 0.962/0.726/0.433`. stattx_medium_guard raw.
- `reduced_feature_auto_topk20` original_test_all_10s+: acc `0.913531`, recalls 0.973/0.885/0.696`. reduced tabular model; not waveform-only.

## Candidates

- `multiscale_statpatch_balanced` (search): best_epoch=3, useful_gate=pass, original_report_ran=True, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_geometry_student\N17043_gm_probe\search\multiscale_statpatch_balanced\ckpt_best.pt`
- `multiscale_statpatch_medium_badguard` (search): best_epoch=5, useful_gate=pass, original_report_ran=True, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_geometry_student\N17043_gm_probe\search\multiscale_statpatch_medium_badguard\ckpt_best.pt`

## Notes

- Selection uses synthetic/node diagnostic only.
- BUT buckets are emitted only for useful-gate candidates unless `--but-report-mode always` is used.
- Diagnostic artifacts are saved beside each checkpoint: teacher recovery CSV, error rows, embedding PCA, and waveform mistake panels.
- Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_geometry_student_search_metrics.csv`
