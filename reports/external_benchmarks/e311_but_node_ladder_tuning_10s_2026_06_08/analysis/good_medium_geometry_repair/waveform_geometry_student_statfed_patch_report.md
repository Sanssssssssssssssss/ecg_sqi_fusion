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
 M reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/waveform_geometry_student_smoke_report.md
 M reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/node_ladder_diagnostic_metrics.csv
 M reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/node_promotion_decisions.csv
 M reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/node_registry.csv
 M reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/node_registry.json
 M
```

## Metrics

| Candidate | Run | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | Teacher MAE | Core MAE |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| statfed_patch_tx | search | synthetic_val | 0.995928 | 0.995043 | 0.995086 | 0.999053 | 0.984375 | 2 | 1 | 4 | 0.7647 | 0.6419 |
| statfed_patch_tx | search | synthetic_test | 0.996412 | 0.995217 | 0.993724 | 1.000000 | 0.983402 | 3 | 0 | 4 | 0.7555 | 0.6181 |
| statfed_patch_tx_badcal | search | synthetic_test | 0.996412 | 0.995217 | 0.993724 | 1.000000 | 0.983402 | 3 | 0 | 4 | nan | nan |
| statfed_patch_tx | search | original_test_all_10s+ | 0.775510 | 0.667242 | 0.897527 | 0.721871 | 0.272506 | 373 | 1231 | 53 | nan | nan |
| statfed_patch_tx_badcal | search | original_test_all_10s+ | 0.775510 | 0.667242 | 0.897527 | 0.721871 | 0.272506 | 373 | 1231 | 53 | nan | nan |
| statfed_patch_tx | search | original_all_10s+ | 0.804102 | 0.835879 | 0.730505 | 0.858863 | 0.931315 | 4593 | 1499 | 115 | nan | nan |
| statfed_patch_tx_badcal | search | original_all_10s+ | 0.804102 | 0.835879 | 0.730505 | 0.858863 | 0.931315 | 4593 | 1499 | 115 | nan | nan |
| statfed_patch_tx | search | bad_core_nearboundary | 0.941176 | 0.323232 | 0.000000 | 0.000000 | 0.941176 | 0 | 0 | 7 | nan | nan |
| statfed_patch_tx_badcal | search | bad_core_nearboundary | 0.941176 | 0.323232 | 0.000000 | 0.000000 | 0.941176 | 0 | 0 | 7 | nan | nan |
| statfed_patch_tx | search | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 46 | nan | nan |
| statfed_patch_tx_badcal | search | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 46 | nan | nan |
| statfed_patch_atlas_tx | search | synthetic_val | 0.994183 | 0.992836 | 0.995086 | 0.998106 | 0.976562 | 2 | 2 | 6 | 0.7615 | 0.6490 |
| statfed_patch_atlas_tx | search | synthetic_test | 0.995387 | 0.993892 | 0.991632 | 1.000000 | 0.979253 | 4 | 0 | 5 | 0.7607 | 0.6211 |
| statfed_patch_atlas_tx_badcal | search | synthetic_test | 0.995900 | 0.994731 | 0.991632 | 1.000000 | 0.983402 | 4 | 0 | 4 | nan | nan |
| statfed_patch_atlas_tx | search | original_test_all_10s+ | 0.782706 | 0.674870 | 0.875824 | 0.752824 | 0.279805 | 452 | 1094 | 78 | nan | nan |
| statfed_patch_atlas_tx_badcal | search | original_test_all_10s+ | 0.783178 | 0.678930 | 0.875824 | 0.752824 | 0.289538 | 452 | 1094 | 74 | nan | nan |
| statfed_patch_atlas_tx | search | original_all_10s+ | 0.791510 | 0.826727 | 0.695241 | 0.876176 | 0.931693 | 5194 | 1314 | 140 | nan | nan |
| statfed_patch_atlas_tx_badcal | search | original_all_10s+ | 0.791601 | 0.826843 | 0.695241 | 0.875894 | 0.932829 | 5194 | 1314 | 135 | nan | nan |
| statfed_patch_atlas_tx | search | bad_core_nearboundary | 0.966387 | 0.327635 | 0.000000 | 0.000000 | 0.966387 | 0 | 0 | 4 | nan | nan |
| statfed_patch_atlas_tx_badcal | search | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | nan | nan |
| statfed_patch_atlas_tx | search | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 74 | nan | nan |
| statfed_patch_atlas_tx_badcal | search | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 74 | nan | nan |

## Baselines

- `47_feature_tabular_upper_bound` original_test_all_10s+: acc `0.963548`, recalls 0.956/0.973/0.927`. tabular-only upper bound; geometry features are inference inputs.
- `current_best_waveform_transformer` original_test_all_10s+: acc `0.811018`, recalls 0.961/0.726/0.401`. aug_convtx_balanced_focal raw.
- `stattoken_baseline` original_test_all_10s+: acc `0.813141`, recalls 0.962/0.726/0.433`. stattx_medium_guard raw.
- `reduced_feature_auto_topk20` original_test_all_10s+: acc `0.913531`, recalls 0.973/0.885/0.696`. reduced tabular model; not waveform-only.

## Candidates

- `statfed_patch_tx` (search): best_epoch=7, useful_gate=pass, original_report_ran=True, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_geometry_student\N17043_gm_probe\search\statfed_patch_tx\ckpt_best.pt`
- `statfed_patch_atlas_tx` (search): best_epoch=7, useful_gate=pass, original_report_ran=True, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_geometry_student\N17043_gm_probe\search\statfed_patch_atlas_tx\ckpt_best.pt`

## Notes

- Selection uses synthetic/node diagnostic only.
- BUT buckets are emitted only for useful-gate candidates unless `--but-report-mode always` is used.
- Diagnostic artifacts are saved beside each checkpoint: teacher recovery CSV, error rows, embedding PCA, and waveform mistake panels.
- Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_geometry_student_search_metrics.csv`
