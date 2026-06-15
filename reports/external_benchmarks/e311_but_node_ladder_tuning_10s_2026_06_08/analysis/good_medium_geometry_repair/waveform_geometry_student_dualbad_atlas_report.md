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
| statfed_patch_dualbad_atlas | search | synthetic_val | 0.993601 | 0.992260 | 0.987715 | 1.000000 | 0.976562 | 5 | 0 | 6 | 0.7541 | 0.6657 |
| statfed_patch_dualbad_atlas | search | synthetic_test | 0.991799 | 0.990475 | 0.976987 | 1.000000 | 0.979253 | 11 | 0 | 5 | 0.7430 | 0.6289 |
| statfed_patch_dualbad_atlas_badcal | search | synthetic_test | 0.991799 | 0.990475 | 0.976987 | 1.000000 | 0.979253 | 11 | 0 | 5 | nan | nan |
| statfed_patch_dualbad_atlas | search | original_test_all_10s+ | 0.813731 | 0.699560 | 0.856593 | 0.826932 | 0.291971 | 522 | 763 | 103 | nan | nan |
| statfed_patch_dualbad_atlas_badcal | search | original_test_all_10s+ | 0.812434 | 0.696639 | 0.856319 | 0.823995 | 0.299270 | 520 | 759 | 101 | nan | nan |
| statfed_patch_dualbad_atlas | search | original_all_10s+ | 0.780799 | 0.819264 | 0.648301 | 0.917858 | 0.932450 | 5994 | 870 | 168 | nan | nan |
| statfed_patch_dualbad_atlas_badcal | search | original_all_10s+ | 0.780252 | 0.818176 | 0.648184 | 0.915506 | 0.934153 | 5989 | 866 | 160 | nan | nan |
| statfed_patch_dualbad_atlas | search | bad_core_nearboundary | 0.991597 | 0.331927 | 0.000000 | 0.000000 | 0.991597 | 0 | 0 | 1 | nan | nan |
| statfed_patch_dualbad_atlas_badcal | search | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | nan | nan |
| statfed_patch_dualbad_atlas | search | bad_outlier_stress | 0.006849 | 0.004535 | 0.000000 | 0.000000 | 0.006849 | 0 | 0 | 102 | nan | nan |
| statfed_patch_dualbad_atlas_badcal | search | bad_outlier_stress | 0.013699 | 0.009009 | 0.000000 | 0.000000 | 0.013699 | 0 | 0 | 101 | nan | nan |
| multiscale_statpatch_dualbad_atlas | search | synthetic_val | 0.994764 | 0.993402 | 0.995086 | 0.999053 | 0.976562 | 2 | 1 | 6 | 0.7717 | 0.6539 |
| multiscale_statpatch_dualbad_atlas | search | synthetic_test | 0.994874 | 0.993406 | 0.989540 | 1.000000 | 0.979253 | 5 | 0 | 5 | 0.7783 | 0.6198 |
| multiscale_statpatch_dualbad_atlas_badcal | search | synthetic_test | 0.993849 | 0.991758 | 0.989540 | 0.998377 | 0.979253 | 5 | 0 | 5 | nan | nan |
| multiscale_statpatch_dualbad_atlas | search | original_test_all_10s+ | 0.792025 | 0.683540 | 0.854945 | 0.785133 | 0.309002 | 521 | 923 | 63 | nan | nan |
| multiscale_statpatch_dualbad_atlas_badcal | search | original_test_all_10s+ | 0.789666 | 0.676690 | 0.854396 | 0.779937 | 0.321168 | 521 | 903 | 60 | nan | nan |
| multiscale_statpatch_dualbad_atlas | search | original_all_10s+ | 0.798428 | 0.831404 | 0.699583 | 0.889255 | 0.934532 | 5106 | 1136 | 124 | nan | nan |
| multiscale_statpatch_dualbad_atlas_badcal | search | original_all_10s+ | 0.797579 | 0.829456 | 0.699231 | 0.886244 | 0.936424 | 5099 | 1116 | 116 | nan | nan |
| multiscale_statpatch_dualbad_atlas | search | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | nan | nan |
| multiscale_statpatch_dualbad_atlas_badcal | search | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | nan | nan |
| multiscale_statpatch_dualbad_atlas | search | bad_outlier_stress | 0.027397 | 0.017778 | 0.000000 | 0.000000 | 0.027397 | 0 | 0 | 63 | nan | nan |
| multiscale_statpatch_dualbad_atlas_badcal | search | bad_outlier_stress | 0.044521 | 0.028415 | 0.000000 | 0.000000 | 0.044521 | 0 | 0 | 60 | nan | nan |

## Baselines

- `47_feature_tabular_upper_bound` original_test_all_10s+: acc `0.963548`, recalls 0.956/0.973/0.927`. tabular-only upper bound; geometry features are inference inputs.
- `current_best_waveform_transformer` original_test_all_10s+: acc `0.811018`, recalls 0.961/0.726/0.401`. aug_convtx_balanced_focal raw.
- `stattoken_baseline` original_test_all_10s+: acc `0.813141`, recalls 0.962/0.726/0.433`. stattx_medium_guard raw.
- `reduced_feature_auto_topk20` original_test_all_10s+: acc `0.913531`, recalls 0.973/0.885/0.696`. reduced tabular model; not waveform-only.

## Candidates

- `statfed_patch_dualbad_atlas` (search): best_epoch=7, useful_gate=pass, original_report_ran=True, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_geometry_student\N17043_gm_probe\search\statfed_patch_dualbad_atlas\ckpt_best.pt`
- `multiscale_statpatch_dualbad_atlas` (search): best_epoch=8, useful_gate=pass, original_report_ran=True, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_geometry_student\N17043_gm_probe\search\multiscale_statpatch_dualbad_atlas\ckpt_best.pt`

## Notes

- Selection uses synthetic/node diagnostic only.
- BUT buckets are emitted only for useful-gate candidates unless `--but-report-mode always` is used.
- Diagnostic artifacts are saved beside each checkpoint: teacher recovery CSV, error rows, embedding PCA, and waveform mistake panels.
- Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_geometry_student_search_metrics.csv`
