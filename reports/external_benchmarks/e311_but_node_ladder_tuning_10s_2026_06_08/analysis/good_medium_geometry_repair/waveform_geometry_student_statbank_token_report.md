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
| statbank_token_balanced | search | synthetic_val | 0.992437 | 0.991122 | 0.985258 | 0.999053 | 0.976562 | 6 | 1 | 6 | 0.7369 | 0.6486 |
| statbank_token_balanced | search | synthetic_test | 0.992312 | 0.991000 | 0.991632 | 0.995130 | 0.979253 | 4 | 6 | 5 | 0.7424 | 0.6222 |
| statbank_token_balanced_badcal | search | synthetic_test | 0.992312 | 0.991000 | 0.991632 | 0.995130 | 0.979253 | 4 | 6 | 5 | nan | nan |
| statbank_token_balanced | search | original_test_all_10s+ | 0.766073 | 0.662647 | 0.875824 | 0.721193 | 0.277372 | 452 | 1234 | 32 | nan | nan |
| statbank_token_balanced_badcal | search | original_test_all_10s+ | 0.764539 | 0.660990 | 0.875824 | 0.717126 | 0.289538 | 452 | 1230 | 27 | nan | nan |
| statbank_token_balanced | search | original_all_10s+ | 0.811142 | 0.840934 | 0.751159 | 0.847384 | 0.931693 | 4240 | 1621 | 95 | nan | nan |
| statbank_token_balanced_badcal | search | original_all_10s+ | 0.810475 | 0.839571 | 0.750983 | 0.843432 | 0.936045 | 4238 | 1614 | 72 | nan | nan |
| statbank_token_balanced | search | bad_core_nearboundary | 0.957983 | 0.326180 | 0.000000 | 0.000000 | 0.957983 | 0 | 0 | 5 | nan | nan |
| statbank_token_balanced_badcal | search | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | nan | nan |
| statbank_token_balanced | search | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 27 | nan | nan |
| statbank_token_balanced_badcal | search | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 27 | nan | nan |
| statbank_token_badguard | search | synthetic_val | 0.993601 | 0.992264 | 0.990172 | 0.999053 | 0.976562 | 4 | 1 | 6 | 0.7485 | 0.6508 |
| statbank_token_badguard | search | synthetic_test | 0.991799 | 0.990514 | 0.989540 | 0.995130 | 0.979253 | 5 | 6 | 5 | 0.7424 | 0.6269 |
| statbank_token_badguard_badcal | search | synthetic_test | 0.991799 | 0.990514 | 0.989540 | 0.995130 | 0.979253 | 5 | 6 | 5 | nan | nan |
| statbank_token_badguard | search | original_test_all_10s+ | 0.752153 | 0.638128 | 0.896429 | 0.678717 | 0.265207 | 376 | 1376 | 30 | nan | nan |
| statbank_token_badguard_badcal | search | original_test_all_10s+ | 0.748732 | 0.640154 | 0.895604 | 0.668098 | 0.316302 | 374 | 1357 | 15 | nan | nan |
| statbank_token_badguard | search | original_all_10s+ | 0.811112 | 0.837945 | 0.765358 | 0.824991 | 0.930747 | 3959 | 1800 | 93 | nan | nan |
| statbank_token_badguard_badcal | search | original_all_10s+ | 0.808988 | 0.833681 | 0.763481 | 0.818875 | 0.935856 | 3944 | 1771 | 72 | nan | nan |
| statbank_token_badguard | search | bad_core_nearboundary | 0.907563 | 0.317181 | 0.000000 | 0.000000 | 0.907563 | 0 | 0 | 11 | nan | nan |
| statbank_token_badguard_badcal | search | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | nan | nan |
| statbank_token_badguard | search | bad_outlier_stress | 0.003425 | 0.002275 | 0.000000 | 0.000000 | 0.003425 | 0 | 0 | 19 | nan | nan |
| statbank_token_badguard_badcal | search | bad_outlier_stress | 0.037671 | 0.024202 | 0.000000 | 0.000000 | 0.037671 | 0 | 0 | 15 | nan | nan |

## Baselines

- `47_feature_tabular_upper_bound` original_test_all_10s+: acc `0.963548`, recalls 0.956/0.973/0.927`. tabular-only upper bound; geometry features are inference inputs.
- `current_best_waveform_transformer` original_test_all_10s+: acc `0.811018`, recalls 0.961/0.726/0.401`. aug_convtx_balanced_focal raw.
- `stattoken_baseline` original_test_all_10s+: acc `0.813141`, recalls 0.962/0.726/0.433`. stattx_medium_guard raw.
- `reduced_feature_auto_topk20` original_test_all_10s+: acc `0.913531`, recalls 0.973/0.885/0.696`. reduced tabular model; not waveform-only.

## Candidates

- `statbank_token_balanced` (search): best_epoch=4, useful_gate=pass, original_report_ran=True, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_geometry_student\N17043_gm_probe\search\statbank_token_balanced\ckpt_best.pt`
- `statbank_token_badguard` (search): best_epoch=6, useful_gate=pass, original_report_ran=True, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_geometry_student\N17043_gm_probe\search\statbank_token_badguard\ckpt_best.pt`

## Notes

- Selection uses synthetic/node diagnostic only.
- BUT buckets are emitted only for useful-gate candidates unless `--but-report-mode always` is used.
- Diagnostic artifacts are saved beside each checkpoint: teacher recovery CSV, error rows, embedding PCA, and waveform mistake panels.
- Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_geometry_student_search_metrics.csv`
