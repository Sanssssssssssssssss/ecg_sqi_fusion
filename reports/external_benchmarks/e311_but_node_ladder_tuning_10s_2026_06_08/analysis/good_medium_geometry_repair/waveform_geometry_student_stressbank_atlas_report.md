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
| stressbank_statfed_atlas | search | synthetic_val | 0.994183 | 0.992836 | 0.995086 | 0.998106 | 0.976562 | 2 | 2 | 6 | 0.7633 | 0.6518 |
| stressbank_statfed_atlas | search | synthetic_test | 0.993849 | 0.992447 | 0.993724 | 0.996753 | 0.979253 | 3 | 4 | 5 | 0.7461 | 0.6218 |
| stressbank_statfed_atlas_badcal | search | synthetic_test | 0.993849 | 0.992447 | 0.993724 | 0.996753 | 0.979253 | 3 | 4 | 5 | nan | nan |
| stressbank_statfed_atlas | search | original_test_all_10s+ | 0.784004 | 0.678817 | 0.909066 | 0.726841 | 0.291971 | 330 | 1203 | 69 | nan | nan |
| stressbank_statfed_atlas_badcal | search | original_test_all_10s+ | 0.782352 | 0.672516 | 0.908242 | 0.723452 | 0.301703 | 328 | 1185 | 68 | nan | nan |
| stressbank_statfed_atlas | search | original_all_10s+ | 0.846341 | 0.867367 | 0.819926 | 0.845879 | 0.932450 | 3067 | 1627 | 133 | nan | nan |
| stressbank_statfed_atlas_badcal | search | original_all_10s+ | 0.845400 | 0.865205 | 0.819163 | 0.843338 | 0.934153 | 3060 | 1603 | 128 | nan | nan |
| stressbank_statfed_atlas | search | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | nan | nan |
| stressbank_statfed_atlas_badcal | search | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | nan | nan |
| stressbank_statfed_atlas | search | bad_outlier_stress | 0.003425 | 0.002275 | 0.000000 | 0.000000 | 0.003425 | 0 | 0 | 69 | nan | nan |
| stressbank_statfed_atlas_badcal | search | bad_outlier_stress | 0.017123 | 0.011223 | 0.000000 | 0.000000 | 0.017123 | 0 | 0 | 68 | nan | nan |
| stressbank_multiscale_atlas | search | synthetic_val | 0.993601 | 0.992260 | 0.987715 | 1.000000 | 0.976562 | 5 | 0 | 6 | 0.7536 | 0.6511 |
| stressbank_multiscale_atlas | search | synthetic_test | 0.993849 | 0.992432 | 0.985356 | 1.000000 | 0.979253 | 7 | 0 | 5 | 0.7573 | 0.6195 |
| stressbank_multiscale_atlas_badcal | search | synthetic_test | 0.993849 | 0.992432 | 0.985356 | 1.000000 | 0.979253 | 7 | 0 | 5 | nan | nan |
| stressbank_multiscale_atlas | search | original_test_all_10s+ | 0.791200 | 0.683543 | 0.841484 | 0.795978 | 0.294404 | 577 | 894 | 59 | nan | nan |
| stressbank_multiscale_atlas_badcal | search | original_test_all_10s+ | 0.790846 | 0.683040 | 0.840385 | 0.793945 | 0.318735 | 577 | 868 | 57 | nan | nan |
| stressbank_multiscale_atlas | search | original_all_10s+ | 0.748361 | 0.793731 | 0.595787 | 0.901016 | 0.933396 | 6888 | 1036 | 119 | nan | nan |
| stressbank_multiscale_atlas_badcal | search | original_all_10s+ | 0.747785 | 0.792123 | 0.595142 | 0.899134 | 0.935667 | 6886 | 1009 | 115 | nan | nan |
| stressbank_multiscale_atlas | search | bad_core_nearboundary | 0.991597 | 0.331927 | 0.000000 | 0.000000 | 0.991597 | 0 | 0 | 1 | nan | nan |
| stressbank_multiscale_atlas_badcal | search | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | nan | nan |
| stressbank_multiscale_atlas | search | bad_outlier_stress | 0.010274 | 0.006780 | 0.000000 | 0.000000 | 0.010274 | 0 | 0 | 58 | nan | nan |
| stressbank_multiscale_atlas_badcal | search | bad_outlier_stress | 0.041096 | 0.026316 | 0.000000 | 0.000000 | 0.041096 | 0 | 0 | 57 | nan | nan |

## Baselines

- `47_feature_tabular_upper_bound` original_test_all_10s+: acc `0.963548`, recalls 0.956/0.973/0.927`. tabular-only upper bound; geometry features are inference inputs.
- `current_best_waveform_transformer` original_test_all_10s+: acc `0.811018`, recalls 0.961/0.726/0.401`. aug_convtx_balanced_focal raw.
- `stattoken_baseline` original_test_all_10s+: acc `0.813141`, recalls 0.962/0.726/0.433`. stattx_medium_guard raw.
- `reduced_feature_auto_topk20` original_test_all_10s+: acc `0.913531`, recalls 0.973/0.885/0.696`. reduced tabular model; not waveform-only.

## Candidates

- `stressbank_statfed_atlas` (search): best_epoch=4, useful_gate=pass, original_report_ran=True, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_geometry_student\N17043_gm_probe\search\stressbank_statfed_atlas\ckpt_best.pt`
- `stressbank_multiscale_atlas` (search): best_epoch=8, useful_gate=pass, original_report_ran=True, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_geometry_student\N17043_gm_probe\search\stressbank_multiscale_atlas\ckpt_best.pt`

## Notes

- Selection uses synthetic/node diagnostic only.
- BUT buckets are emitted only for useful-gate candidates unless `--but-report-mode always` is used.
- Diagnostic artifacts are saved beside each checkpoint: teacher recovery CSV, error rows, embedding PCA, and waveform mistake panels.
- Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_geometry_student_search_metrics.csv`
