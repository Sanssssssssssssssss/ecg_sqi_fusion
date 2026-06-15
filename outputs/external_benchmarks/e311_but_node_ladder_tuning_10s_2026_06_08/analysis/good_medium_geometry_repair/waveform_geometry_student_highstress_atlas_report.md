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
| stressbank_statfed_highstress | search | synthetic_val | 0.993601 | 0.992264 | 0.990172 | 0.999053 | 0.976562 | 4 | 1 | 6 | 0.7429 | 0.6524 |
| stressbank_statfed_highstress | search | synthetic_test | 0.992824 | 0.991460 | 0.983264 | 0.999188 | 0.979253 | 8 | 1 | 5 | 0.7414 | 0.6273 |
| stressbank_statfed_highstress_badcal | search | synthetic_test | 0.992824 | 0.991460 | 0.983264 | 0.999188 | 0.979253 | 8 | 1 | 5 | nan | nan |
| stressbank_statfed_highstress | search | original_test_all_10s+ | 0.763714 | 0.646276 | 0.893956 | 0.704925 | 0.243309 | 386 | 1301 | 75 | nan | nan |
| stressbank_statfed_highstress_badcal | search | original_test_all_10s+ | 0.763950 | 0.650250 | 0.890934 | 0.701988 | 0.306569 | 384 | 1229 | 53 | nan | nan |
| stressbank_statfed_highstress | search | original_all_10s+ | 0.791935 | 0.825547 | 0.710380 | 0.855947 | 0.926206 | 4932 | 1514 | 153 | nan | nan |
| stressbank_statfed_highstress_badcal | search | original_all_10s+ | 0.790296 | 0.820977 | 0.706272 | 0.853312 | 0.934532 | 4928 | 1425 | 114 | nan | nan |
| stressbank_statfed_highstress | search | bad_core_nearboundary | 0.840336 | 0.304414 | 0.000000 | 0.000000 | 0.840336 | 0 | 0 | 19 | nan | nan |
| stressbank_statfed_highstress_badcal | search | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | nan | nan |
| stressbank_statfed_highstress | search | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 56 | nan | nan |
| stressbank_statfed_highstress_badcal | search | bad_outlier_stress | 0.023973 | 0.015608 | 0.000000 | 0.000000 | 0.023973 | 0 | 0 | 53 | nan | nan |
| stressbank_multiscale_highstress | search | synthetic_val | 0.994183 | 0.992833 | 0.992629 | 0.999053 | 0.976562 | 3 | 1 | 6 | 0.7537 | 0.6598 |
| stressbank_multiscale_highstress | search | synthetic_test | 0.993337 | 0.991961 | 0.991632 | 0.996753 | 0.979253 | 4 | 4 | 5 | 0.7517 | 0.6243 |
| stressbank_multiscale_highstress_badcal | search | synthetic_test | 0.993337 | 0.991961 | 0.991632 | 0.996753 | 0.979253 | 4 | 4 | 5 | nan | nan |
| stressbank_multiscale_highstress | search | original_test_all_10s+ | 0.796980 | 0.658180 | 0.907418 | 0.759602 | 0.221411 | 336 | 1051 | 114 | nan | nan |
| stressbank_multiscale_highstress_badcal | search | original_test_all_10s+ | 0.793677 | 0.672154 | 0.907418 | 0.746046 | 0.299270 | 329 | 1050 | 82 | nan | nan |
| stressbank_multiscale_highstress | search | original_all_10s+ | 0.808775 | 0.838840 | 0.728334 | 0.879469 | 0.926017 | 4626 | 1263 | 183 | nan | nan |
| stressbank_multiscale_highstress_badcal | search | original_all_10s+ | 0.807926 | 0.836832 | 0.728334 | 0.872883 | 0.933964 | 4603 | 1262 | 142 | nan | nan |
| stressbank_multiscale_highstress | search | bad_core_nearboundary | 0.756303 | 0.287081 | 0.000000 | 0.000000 | 0.756303 | 0 | 0 | 29 | nan | nan |
| stressbank_multiscale_highstress_badcal | search | bad_core_nearboundary | 0.983193 | 0.330508 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 2 | nan | nan |
| stressbank_multiscale_highstress | search | bad_outlier_stress | 0.003425 | 0.002275 | 0.000000 | 0.000000 | 0.003425 | 0 | 0 | 85 | nan | nan |
| stressbank_multiscale_highstress_badcal | search | bad_outlier_stress | 0.020548 | 0.013423 | 0.000000 | 0.000000 | 0.020548 | 0 | 0 | 80 | nan | nan |

## Baselines

- `47_feature_tabular_upper_bound` original_test_all_10s+: acc `0.963548`, recalls 0.956/0.973/0.927`. tabular-only upper bound; geometry features are inference inputs.
- `current_best_waveform_transformer` original_test_all_10s+: acc `0.811018`, recalls 0.961/0.726/0.401`. aug_convtx_balanced_focal raw.
- `stattoken_baseline` original_test_all_10s+: acc `0.813141`, recalls 0.962/0.726/0.433`. stattx_medium_guard raw.
- `reduced_feature_auto_topk20` original_test_all_10s+: acc `0.913531`, recalls 0.973/0.885/0.696`. reduced tabular model; not waveform-only.

## Candidates

- `stressbank_statfed_highstress` (search): best_epoch=3, useful_gate=pass, original_report_ran=True, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_geometry_student\N17043_gm_probe\search\stressbank_statfed_highstress\ckpt_best.pt`
- `stressbank_multiscale_highstress` (search): best_epoch=4, useful_gate=pass, original_report_ran=True, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_geometry_student\N17043_gm_probe\search\stressbank_multiscale_highstress\ckpt_best.pt`

## Notes

- Selection uses synthetic/node diagnostic only.
- BUT buckets are emitted only for useful-gate candidates unless `--but-report-mode always` is used.
- Diagnostic artifacts are saved beside each checkpoint: teacher recovery CSV, error rows, embedding PCA, and waveform mistake panels.
- Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_geometry_student_search_metrics.csv`
