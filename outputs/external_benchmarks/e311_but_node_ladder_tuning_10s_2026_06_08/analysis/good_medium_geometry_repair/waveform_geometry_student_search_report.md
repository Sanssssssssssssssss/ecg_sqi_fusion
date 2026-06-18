# Waveform Geometry Student

Input is waveform only. SQI/geometry features are teacher targets in the loss, not classifier inputs. Original BUT is report-only.

## Dirty Worktree Warning

Existing worktree changes were present before this runner. This experiment writes only external analysis/report/run outputs.

```text
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
 M reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/waveform_transformer_augmented_original_report.md
 M reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/waveform_transformer_original_adaptation_report.md
 M reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/waveform_transformer_search_report.md
 M reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/node_ladder_diagnostic_metrics.csv
 M reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/node_promotion_decisions.csv
 M reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/node_registry.csv
 M reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/node_registry.json
 M src/sqi_pipeline/qrs/paper_detectors.py
 M src/transformer_pipeline/e311_uformer_data.py
 M src/transformer_pipeline/external_benchmarks/but_bad_boundary_tuning.py
 M src/transformer_pipeline/external_benchmarks/run.py
 M src/transformer_pipeline/models/uformer1d.py
 M src/transformer_pipeline/train_uformer_mainline.py
?? reports/external_benchmarks/e311_but_big_uformer_long_search_10s_2026_06_06/
?? reports/external_benchmarks/e311_but_boundary_head_adaptation_10s_2026_06_03/
?? reports/external_benchmarks/e311_but_clean_core_ta
```

## Metrics

| Candidate | Run | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | Teacher MAE | Core MAE |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| record111_detbase_frozenbridge_artifact_a050 | search | synthetic_val | 0.993019 | 0.991698 | 0.990172 | 0.998106 | 0.976562 | 4 | 2 | 6 | 0.6653 | 0.6650 |
| record111_detbase_frozenbridge_artifact_a050 | search | synthetic_test | 0.994874 | 0.993411 | 0.993724 | 0.998377 | 0.979253 | 3 | 2 | 5 | 0.6975 | 0.6248 |
| record111_detbase_frozenbridge_artifact_a050_badcal | search | synthetic_test | 0.994874 | 0.993411 | 0.993724 | 0.998377 | 0.979253 | 3 | 2 | 5 | nan | nan |
| record111_detbase_frozenbridge_artifact_a050 | search | original_test_all_10s+ | 0.814675 | 0.662657 | 0.900275 | 0.799367 | 0.221411 | 358 | 836 | 142 | nan | nan |
| record111_detbase_frozenbridge_artifact_a050_badcal | search | original_test_all_10s+ | 0.814911 | 0.665495 | 0.900275 | 0.799141 | 0.228710 | 358 | 836 | 141 | nan | nan |
| record111_detbase_frozenbridge_artifact_a050 | search | original_all_10s+ | 0.852682 | 0.870873 | 0.820865 | 0.867049 | 0.926395 | 3045 | 1354 | 206 | nan | nan |
| record111_detbase_frozenbridge_artifact_a050_badcal | search | original_all_10s+ | 0.852743 | 0.870956 | 0.820865 | 0.866955 | 0.926963 | 3045 | 1354 | 205 | nan | nan |
| record111_detbase_frozenbridge_artifact_a050 | search | bad_core_nearboundary | 0.327731 | 0.164557 | 0.000000 | 0.000000 | 0.327731 | 0 | 0 | 80 | nan | nan |
| record111_detbase_frozenbridge_artifact_a050_badcal | search | bad_core_nearboundary | 0.327731 | 0.164557 | 0.000000 | 0.000000 | 0.327731 | 0 | 0 | 80 | nan | nan |
| record111_detbase_frozenbridge_artifact_a050 | search | bad_outlier_stress | 0.178082 | 0.100775 | 0.000000 | 0.000000 | 0.178082 | 0 | 0 | 62 | nan | nan |
| record111_detbase_frozenbridge_artifact_a050_badcal | search | bad_outlier_stress | 0.188356 | 0.105668 | 0.000000 | 0.000000 | 0.188356 | 0 | 0 | 61 | nan | nan |

## Baselines

- `47_feature_tabular_upper_bound` original_test_all_10s+: acc `0.963548`, recalls 0.956/0.973/0.927`. tabular-only upper bound; geometry features are inference inputs.
- `current_best_waveform_transformer` original_test_all_10s+: acc `0.811018`, recalls 0.961/0.726/0.401`. aug_convtx_balanced_focal raw.
- `stattoken_baseline` original_test_all_10s+: acc `0.813141`, recalls 0.962/0.726/0.433`. stattx_medium_guard raw.
- `reduced_feature_auto_topk20` original_test_all_10s+: acc `0.913531`, recalls 0.973/0.885/0.696`. reduced tabular model; not waveform-only.

## Candidates

- `record111_detbase_frozenbridge_artifact_a050` (search): best_epoch=3, useful_gate=pass, original_report_ran=True, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_geometry_student\N17043_gm_probe\search\record111_detbase_frozenbridge_artifact_a050\ckpt_best.pt`

## Notes

- Selection uses synthetic/node diagnostic only.
- BUT buckets are emitted only for useful-gate candidates unless `--but-report-mode always` is used.
- Diagnostic artifacts are saved beside each checkpoint: teacher recovery CSV, error rows, embedding PCA, and waveform mistake panels.
- Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_geometry_student_search_metrics.csv`
