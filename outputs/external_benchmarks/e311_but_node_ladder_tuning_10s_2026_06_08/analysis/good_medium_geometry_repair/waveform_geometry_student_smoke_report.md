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
| mapstudent_patchtx | smoke | synthetic_val | 0.902269 | 0.907748 | 0.896806 | 0.886364 | 0.976562 | 42 | 120 | 4 | 0.7618 | 0.6494 |
| mapstudent_patchtx | smoke | synthetic_test | 0.935418 | 0.938414 | 0.905858 | 0.938312 | 0.979253 | 45 | 76 | 5 | 0.7530 | 0.6256 |
| mapstudent_patchtx_badcal | smoke | synthetic_test | 0.934905 | 0.937580 | 0.905858 | 0.937500 | 0.979253 | 45 | 76 | 5 | nan | nan |
| masked_geometry_mae | smoke | synthetic_val | 0.896451 | 0.899398 | 0.823096 | 0.905303 | 0.976562 | 72 | 100 | 5 | 0.7648 | 0.6550 |
| masked_geometry_mae | smoke | synthetic_test | 0.915428 | 0.920404 | 0.876569 | 0.918019 | 0.979253 | 59 | 101 | 5 | 0.7581 | 0.6247 |
| masked_geometry_mae_badcal | smoke | synthetic_test | 0.915428 | 0.919842 | 0.876569 | 0.918019 | 0.979253 | 58 | 101 | 5 | nan | nan |
| stattoken_v2 | smoke | synthetic_val | 0.979639 | 0.978310 | 0.928747 | 0.999053 | 0.980469 | 29 | 0 | 5 | 0.7575 | 0.6516 |
| stattoken_v2 | smoke | synthetic_test | 0.980010 | 0.979419 | 0.933054 | 0.997565 | 0.983402 | 32 | 3 | 4 | 0.7443 | 0.6211 |
| stattoken_v2_badcal | smoke | synthetic_test | 0.980010 | 0.979419 | 0.933054 | 0.997565 | 0.983402 | 32 | 3 | 4 | nan | nan |
| morphology_token_tx | smoke | synthetic_val | 0.987202 | 0.986020 | 0.977887 | 0.993371 | 0.976562 | 9 | 7 | 6 | 0.7453 | 0.6410 |
| morphology_token_tx | smoke | synthetic_test | 0.989236 | 0.988044 | 0.972803 | 0.997565 | 0.979253 | 13 | 3 | 5 | 0.7401 | 0.6173 |
| morphology_token_tx_badcal | smoke | synthetic_test | 0.990261 | 0.989718 | 0.972803 | 0.997565 | 0.987552 | 13 | 3 | 3 | nan | nan |

## Baselines

- `47_feature_tabular_upper_bound` original_test_all_10s+: acc `0.963548`, recalls 0.956/0.973/0.927`. tabular-only upper bound; geometry features are inference inputs.
- `current_best_waveform_transformer` original_test_all_10s+: acc `0.811018`, recalls 0.961/0.726/0.401`. aug_convtx_balanced_focal raw.
- `stattoken_baseline` original_test_all_10s+: acc `0.813141`, recalls 0.962/0.726/0.433`. stattx_medium_guard raw.
- `reduced_feature_auto_topk20` original_test_all_10s+: acc `0.913531`, recalls 0.973/0.885/0.696`. reduced tabular model; not waveform-only.

## Candidates

- `mapstudent_patchtx` (smoke): best_epoch=2, useful_gate=fail, original_report_ran=False, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_geometry_student\N17043_gm_probe\smoke\mapstudent_patchtx\ckpt_best.pt`
- `masked_geometry_mae` (smoke): best_epoch=2, useful_gate=fail, original_report_ran=False, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_geometry_student\N17043_gm_probe\smoke\masked_geometry_mae\ckpt_best.pt`
- `stattoken_v2` (smoke): best_epoch=2, useful_gate=pass, original_report_ran=False, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_geometry_student\N17043_gm_probe\smoke\stattoken_v2\ckpt_best.pt`
- `morphology_token_tx` (smoke): best_epoch=1, useful_gate=pass, original_report_ran=False, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_geometry_student\N17043_gm_probe\smoke\morphology_token_tx\ckpt_best.pt`

## Notes

- Selection uses synthetic/node diagnostic only.
- BUT buckets are emitted only for useful-gate candidates unless `--but-report-mode always` is used.
- Diagnostic artifacts are saved beside each checkpoint: teacher recovery CSV, error rows, embedding PCA, and waveform mistake panels.
- Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_geometry_student_smoke_metrics.csv`
