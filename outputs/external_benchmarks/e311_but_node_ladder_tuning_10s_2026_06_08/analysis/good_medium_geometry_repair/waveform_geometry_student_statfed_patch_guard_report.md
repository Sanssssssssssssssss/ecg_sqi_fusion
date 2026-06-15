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
| statfed_patch_medium_guard | search | synthetic_val | 0.993601 | 0.992023 | 0.990172 | 0.999053 | 0.976562 | 4 | 0 | 6 | 0.7652 | 0.6532 |
| statfed_patch_medium_guard | search | synthetic_test | 0.995387 | 0.993892 | 0.991632 | 1.000000 | 0.979253 | 4 | 0 | 5 | 0.7587 | 0.6229 |
| statfed_patch_medium_guard_badcal | search | synthetic_test | 0.995387 | 0.993892 | 0.991632 | 1.000000 | 0.979253 | 4 | 0 | 5 | nan | nan |
| statfed_patch_medium_guard | search | original_test_all_10s+ | 0.804530 | 0.643224 | 0.892582 | 0.791008 | 0.170316 | 391 | 925 | 148 | nan | nan |
| statfed_patch_medium_guard_badcal | search | original_test_all_10s+ | 0.805710 | 0.656941 | 0.892582 | 0.790556 | 0.199513 | 391 | 925 | 136 | nan | nan |
| statfed_patch_medium_guard | search | original_all_10s+ | 0.800704 | 0.832619 | 0.705451 | 0.894806 | 0.918638 | 5020 | 1113 | 222 | nan | nan |
| statfed_patch_medium_guard_badcal | search | original_all_10s+ | 0.800977 | 0.832987 | 0.705451 | 0.894148 | 0.921665 | 5020 | 1113 | 207 | nan | nan |
| statfed_patch_medium_guard | search | bad_core_nearboundary | 0.588235 | 0.246914 | 0.000000 | 0.000000 | 0.588235 | 0 | 0 | 49 | nan | nan |
| statfed_patch_medium_guard_badcal | search | bad_core_nearboundary | 0.689076 | 0.271973 | 0.000000 | 0.000000 | 0.689076 | 0 | 0 | 37 | nan | nan |
| statfed_patch_medium_guard | search | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 99 | nan | nan |
| statfed_patch_medium_guard_badcal | search | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 99 | nan | nan |
| statfed_patch_badstress_guard | search | synthetic_val | 0.994183 | 0.992836 | 0.995086 | 0.998106 | 0.976562 | 2 | 2 | 6 | 0.7514 | 0.6429 |
| statfed_patch_badstress_guard | search | synthetic_test | 0.994874 | 0.993409 | 0.991632 | 0.999188 | 0.979253 | 4 | 1 | 5 | 0.7478 | 0.6110 |
| statfed_patch_badstress_guard_badcal | search | synthetic_test | 0.994874 | 0.993409 | 0.991632 | 0.999188 | 0.979253 | 4 | 1 | 5 | nan | nan |
| statfed_patch_badstress_guard | search | original_test_all_10s+ | 0.760293 | 0.663326 | 0.890934 | 0.696566 | 0.289538 | 397 | 1343 | 55 | nan | nan |
| statfed_patch_badstress_guard_badcal | search | original_test_all_10s+ | 0.760175 | 0.662993 | 0.890934 | 0.696340 | 0.289538 | 397 | 1343 | 55 | nan | nan |
| statfed_patch_badstress_guard | search | original_all_10s+ | 0.786351 | 0.822294 | 0.700816 | 0.851148 | 0.931883 | 5099 | 1582 | 122 | nan | nan |
| statfed_patch_badstress_guard_badcal | search | original_all_10s+ | 0.786382 | 0.822322 | 0.700816 | 0.850866 | 0.932640 | 5099 | 1582 | 118 | nan | nan |
| statfed_patch_badstress_guard | search | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | nan | nan |
| statfed_patch_badstress_guard_badcal | search | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | nan | nan |
| statfed_patch_badstress_guard | search | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 55 | nan | nan |
| statfed_patch_badstress_guard_badcal | search | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 55 | nan | nan |

## Baselines

- `47_feature_tabular_upper_bound` original_test_all_10s+: acc `0.963548`, recalls 0.956/0.973/0.927`. tabular-only upper bound; geometry features are inference inputs.
- `current_best_waveform_transformer` original_test_all_10s+: acc `0.811018`, recalls 0.961/0.726/0.401`. aug_convtx_balanced_focal raw.
- `stattoken_baseline` original_test_all_10s+: acc `0.813141`, recalls 0.962/0.726/0.433`. stattx_medium_guard raw.
- `reduced_feature_auto_topk20` original_test_all_10s+: acc `0.913531`, recalls 0.973/0.885/0.696`. reduced tabular model; not waveform-only.

## Candidates

- `statfed_patch_medium_guard` (search): best_epoch=4, useful_gate=pass, original_report_ran=True, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_geometry_student\N17043_gm_probe\search\statfed_patch_medium_guard\ckpt_best.pt`
- `statfed_patch_badstress_guard` (search): best_epoch=8, useful_gate=pass, original_report_ran=True, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_geometry_student\N17043_gm_probe\search\statfed_patch_badstress_guard\ckpt_best.pt`

## Notes

- Selection uses synthetic/node diagnostic only.
- BUT buckets are emitted only for useful-gate candidates unless `--but-report-mode always` is used.
- Diagnostic artifacts are saved beside each checkpoint: teacher recovery CSV, error rows, embedding PCA, and waveform mistake panels.
- Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_geometry_student_search_metrics.csv`
