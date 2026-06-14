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
?? reports/external_benchmarks/e311_but_clean_label_core_10s_2026_06_06/
?? reports/external_benchmarks/e311_but_clean_target_fit_grid_10s_2026_06_06/
?? reports/external_benchmarks/e311_but_cleanbut_boundary_relaxation_10s_2026_06_07/
?? r
```

## Metrics

| Candidate | Run | Bucket | Acc | Macro-F1 | Good R | Medium R | Bad R | g->m | m->g | b->m | Teacher MAE | Core MAE |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stattoken_v2_badstress | search | synthetic_val | 0.995928 | 0.994536 | 1.000000 | 0.999053 | 0.976562 | 0 | 1 | 6 | 0.7473 | 0.6479 |
| stattoken_v2_badstress | search | synthetic_test | 0.990774 | 0.989539 | 0.985356 | 0.995130 | 0.979253 | 7 | 6 | 5 | 0.7525 | 0.6247 |
| stattoken_v2_badstress_badcal | search | synthetic_test | 0.990774 | 0.989539 | 0.985356 | 0.995130 | 0.979253 | 7 | 6 | 5 | nan | nan |
| stattoken_v2_badstress | search | original_test_all_10s+ | 0.834611 | 0.700971 | 0.824451 | 0.896521 | 0.257908 | 639 | 458 | 208 | nan | nan |
| stattoken_v2_badstress_badcal | search | original_test_all_10s+ | 0.835791 | 0.713408 | 0.824451 | 0.895843 | 0.289538 | 639 | 458 | 195 | nan | nan |
| stattoken_v2_badstress | search | original_all_10s+ | 0.660123 | 0.719073 | 0.399049 | 0.952672 | 0.913718 | 10242 | 503 | 358 | nan | nan |
| stattoken_v2_badstress_badcal | search | original_all_10s+ | 0.662823 | 0.722658 | 0.399049 | 0.951919 | 0.932072 | 10241 | 503 | 261 | nan | nan |
| stattoken_v2_badstress | search | bad_core_nearboundary | 0.890756 | 0.314074 | 0.000000 | 0.000000 | 0.890756 | 0 | 0 | 13 | nan | nan |
| stattoken_v2_badstress_badcal | search | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | nan | nan |
| stattoken_v2_badstress | search | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 195 | nan | nan |
| stattoken_v2_badstress_badcal | search | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 195 | nan | nan |
| morphology_token_badstress | search | synthetic_val | 0.986620 | 0.985579 | 1.000000 | 0.983902 | 0.976562 | 0 | 17 | 6 | 0.7485 | 0.6422 |
| morphology_token_badstress | search | synthetic_test | 0.974885 | 0.973924 | 0.993724 | 0.966721 | 0.979253 | 1 | 41 | 5 | 0.7490 | 0.6157 |
| morphology_token_badstress_badcal | search | synthetic_test | 0.974372 | 0.972889 | 0.991632 | 0.966721 | 0.979253 | 1 | 41 | 5 | nan | nan |
| morphology_token_badstress | search | original_test_all_10s+ | 0.725610 | 0.523878 | 0.929396 | 0.620199 | 0.055961 | 222 | 1531 | 158 | nan | nan |
| morphology_token_badstress_badcal | search | original_test_all_10s+ | 0.723369 | 0.527882 | 0.925000 | 0.617939 | 0.072993 | 222 | 1473 | 156 | nan | nan |
| morphology_token_badstress | search | original_all_10s+ | 0.681424 | 0.486703 | 0.858241 | 0.734475 | 0.004541 | 2352 | 2649 | 5027 | nan | nan |
| morphology_token_badstress_badcal | search | original_all_10s+ | 0.680574 | 0.487928 | 0.856598 | 0.733534 | 0.006433 | 2352 | 2586 | 5023 | nan | nan |
| morphology_token_badstress | search | bad_core_nearboundary | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 119 | nan | nan |
| morphology_token_badstress_badcal | search | bad_core_nearboundary | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 119 | nan | nan |
| morphology_token_badstress | search | bad_outlier_stress | 0.078767 | 0.048677 | 0.000000 | 0.000000 | 0.078767 | 0 | 0 | 39 | nan | nan |
| morphology_token_badstress_badcal | search | bad_outlier_stress | 0.102740 | 0.062112 | 0.000000 | 0.000000 | 0.102740 | 0 | 0 | 37 | nan | nan |

## Baselines

- `47_feature_tabular_upper_bound` original_test_all_10s+: acc `0.963548`, recalls 0.956/0.973/0.927`. tabular-only upper bound; geometry features are inference inputs.
- `current_best_waveform_transformer` original_test_all_10s+: acc `0.811018`, recalls 0.961/0.726/0.401`. aug_convtx_balanced_focal raw.
- `stattoken_baseline` original_test_all_10s+: acc `0.813141`, recalls 0.962/0.726/0.433`. stattx_medium_guard raw.
- `reduced_feature_auto_topk20` original_test_all_10s+: acc `0.913531`, recalls 0.973/0.885/0.696`. reduced tabular model; not waveform-only.

## Candidates

- `stattoken_v2_badstress` (search): best_epoch=6, useful_gate=pass, original_report_ran=True, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_geometry_student\N17043_gm_probe\search\stattoken_v2_badstress\ckpt_best.pt`
- `morphology_token_badstress` (search): best_epoch=8, useful_gate=pass, original_report_ran=True, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_geometry_student\N17043_gm_probe\search\morphology_token_badstress\ckpt_best.pt`

## Notes

- Selection uses synthetic/node diagnostic only.
- BUT buckets are emitted only for useful-gate candidates unless `--but-report-mode always` is used.
- Diagnostic artifacts are saved beside each checkpoint: teacher recovery CSV, error rows, embedding PCA, and waveform mistake panels.
- Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_geometry_student_search_metrics.csv`
