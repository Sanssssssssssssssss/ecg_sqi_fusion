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
 M src/transformer_pipeline/e311_uformer_data.py
 M src/transformer_pipeline/external_benchmarks/but_bad_boundary_tuning.py
 M src/transformer_pipeline/external_benchmarks/run.py
 M src/transformer_pipeline/models/uformer1d.py
 M src/transformer_pipeline/train_uformer_mainline.py
?? reports/external_benchmarks/e311_but_big_uformer_long_search_10s_2026_06_06/
?? reports/external_benchmarks/e311_but_boundary_head_adaptation_10s_2026_06_03/
?? reports/external_benchmarks/e311_but_clean_core_targeted_grid_10s_2026_06_06/
?? reports/exte
```

## Metrics

| Candidate | Run | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | Teacher MAE | Core MAE |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| featurefirst_top20_hardrec_qfeatbin_a050 | search | synthetic_val | 0.993019 | 0.991698 | 0.990172 | 0.998106 | 0.976562 | 4 | 2 | 6 | 0.6533 | 0.6478 |
| featurefirst_top20_hardrec_qfeatbin_a050 | search | synthetic_test | 0.993849 | 0.991884 | 0.989540 | 0.998377 | 0.979253 | 4 | 2 | 5 | 0.6831 | 0.6128 |
| featurefirst_top20_hardrec_qfeatbin_a050_badcal | search | synthetic_test | 0.992824 | 0.989477 | 0.987448 | 0.997565 | 0.979253 | 3 | 2 | 5 | nan | nan |
| featurefirst_top20_hardrec_qfeatbin_a050 | search | original_test_all_10s+ | 0.797806 | 0.625406 | 0.864835 | 0.794171 | 0.243309 | 411 | 553 | 120 | nan | nan |
| featurefirst_top20_hardrec_qfeatbin_a050_badcal | search | original_test_all_10s+ | 0.737761 | 0.610186 | 0.809615 | 0.699503 | 0.513382 | 358 | 336 | 56 | nan | nan |
| featurefirst_top20_hardrec_qfeatbin_a050 | search | original_all_10s+ | 0.826921 | 0.840660 | 0.761192 | 0.882198 | 0.927720 | 3870 | 824 | 190 | nan | nan |
| featurefirst_top20_hardrec_qfeatbin_a050_badcal | search | original_all_10s+ | 0.804376 | 0.804423 | 0.740069 | 0.834776 | 0.950615 | 3691 | 571 | 116 | nan | nan |
| featurefirst_top20_hardrec_qfeatbin_a050 | search | bad_core_nearboundary | 0.596639 | 0.249123 | 0.000000 | 0.000000 | 0.596639 | 0 | 0 | 48 | nan | nan |
| featurefirst_top20_hardrec_qfeatbin_a050_badcal | search | bad_core_nearboundary | 0.941176 | 0.323232 | 0.000000 | 0.000000 | 0.941176 | 0 | 0 | 7 | nan | nan |
| featurefirst_top20_hardrec_qfeatbin_a050 | search | bad_outlier_stress | 0.099315 | 0.060228 | 0.000000 | 0.000000 | 0.099315 | 0 | 0 | 72 | nan | nan |
| featurefirst_top20_hardrec_qfeatbin_a050_badcal | search | bad_outlier_stress | 0.339041 | 0.168798 | 0.000000 | 0.000000 | 0.339041 | 0 | 0 | 49 | nan | nan |
| featurefirst_top20_hardrec_qfeatbin_qrsbase_a050 | search | synthetic_val | 0.991274 | 0.990018 | 0.995086 | 0.993371 | 0.976562 | 2 | 7 | 6 | 0.6511 | 0.6492 |
| featurefirst_top20_hardrec_qfeatbin_qrsbase_a050 | search | synthetic_test | 0.995900 | 0.994379 | 0.995816 | 0.999188 | 0.979253 | 2 | 1 | 5 | 0.6854 | 0.6113 |
| featurefirst_top20_hardrec_qfeatbin_qrsbase_a050_badcal | search | synthetic_test | 0.995387 | 0.993339 | 0.993724 | 0.999188 | 0.979253 | 2 | 1 | 5 | nan | nan |
| featurefirst_top20_hardrec_qfeatbin_qrsbase_a050 | search | original_test_all_10s+ | 0.855963 | 0.712202 | 0.890110 | 0.883868 | 0.253041 | 400 | 506 | 117 | nan | nan |
| featurefirst_top20_hardrec_qfeatbin_qrsbase_a050_badcal | search | original_test_all_10s+ | 0.834729 | 0.688943 | 0.889560 | 0.835743 | 0.338200 | 381 | 504 | 83 | nan | nan |
| featurefirst_top20_hardrec_qfeatbin_qrsbase_a050 | search | original_all_10s+ | 0.865821 | 0.883569 | 0.819809 | 0.908355 | 0.928666 | 3070 | 961 | 185 | nan | nan |
| featurefirst_top20_hardrec_qfeatbin_qrsbase_a050_badcal | search | original_all_10s+ | 0.859570 | 0.871899 | 0.819633 | 0.884927 | 0.937370 | 2973 | 959 | 140 | nan | nan |
| featurefirst_top20_hardrec_qfeatbin_qrsbase_a050 | search | bad_core_nearboundary | 0.865546 | 0.309309 | 0.000000 | 0.000000 | 0.865546 | 0 | 0 | 16 | nan | nan |
| featurefirst_top20_hardrec_qfeatbin_qrsbase_a050_badcal | search | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | nan | nan |
| featurefirst_top20_hardrec_qfeatbin_qrsbase_a050 | search | bad_outlier_stress | 0.003425 | 0.002275 | 0.000000 | 0.000000 | 0.003425 | 0 | 0 | 101 | nan | nan |
| featurefirst_top20_hardrec_qfeatbin_qrsbase_a050_badcal | search | bad_outlier_stress | 0.068493 | 0.042735 | 0.000000 | 0.000000 | 0.068493 | 0 | 0 | 83 | nan | nan |

## Baselines

- `47_feature_tabular_upper_bound` original_test_all_10s+: acc `0.963548`, recalls 0.956/0.973/0.927`. tabular-only upper bound; geometry features are inference inputs.
- `current_best_waveform_transformer` original_test_all_10s+: acc `0.811018`, recalls 0.961/0.726/0.401`. aug_convtx_balanced_focal raw.
- `stattoken_baseline` original_test_all_10s+: acc `0.813141`, recalls 0.962/0.726/0.433`. stattx_medium_guard raw.
- `reduced_feature_auto_topk20` original_test_all_10s+: acc `0.913531`, recalls 0.973/0.885/0.696`. reduced tabular model; not waveform-only.

## Candidates

- `featurefirst_top20_hardrec_qfeatbin_a050` (search): best_epoch=6, useful_gate=pass, original_report_ran=True, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_geometry_student\N17043_gm_probe\search\featurefirst_top20_hardrec_qfeatbin_a050\ckpt_best.pt`
- `featurefirst_top20_hardrec_qfeatbin_qrsbase_a050` (search): best_epoch=11, useful_gate=pass, original_report_ran=True, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_geometry_student\N17043_gm_probe\search\featurefirst_top20_hardrec_qfeatbin_qrsbase_a050\ckpt_best.pt`

## Notes

- Selection uses synthetic/node diagnostic only.
- BUT buckets are emitted only for useful-gate candidates unless `--but-report-mode always` is used.
- Diagnostic artifacts are saved beside each checkpoint: teacher recovery CSV, error rows, embedding PCA, and waveform mistake panels.
- Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_geometry_student_search_metrics.csv`
