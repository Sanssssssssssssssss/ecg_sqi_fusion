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
| physiofact_detrawbeat_localhead_badstress_a050 | report | synthetic_val | 0.955788 | 0.948565 | 0.992629 | 0.936553 | 0.976562 | 3 | 32 | 6 | 0.6712 | 0.6683 |
| physiofact_detrawbeat_localhead_badstress_a050 | report | synthetic_test | 0.976935 | 0.967760 | 0.987448 | 0.971591 | 0.983402 | 5 | 7 | 4 | 0.6977 | 0.6290 |
| physiofact_detrawbeat_localhead_badstress_a050_badcal | report | synthetic_test | 0.976935 | 0.967760 | 0.987448 | 0.971591 | 0.983402 | 5 | 7 | 4 | nan | nan |
| physiofact_detrawbeat_localhead_badstress_a050 | report | original_test_all_10s+ | 0.770320 | 0.670432 | 0.897802 | 0.697921 | 0.420925 | 365 | 1139 | 43 | nan | nan |
| physiofact_detrawbeat_localhead_badstress_a050_badcal | report | original_test_all_10s+ | 0.770202 | 0.670198 | 0.897527 | 0.697921 | 0.420925 | 365 | 1139 | 43 | nan | nan |
| physiofact_detrawbeat_localhead_badstress_a050 | report | original_all_10s+ | 0.838694 | 0.855771 | 0.829314 | 0.801280 | 0.944182 | 2896 | 1868 | 98 | nan | nan |
| physiofact_detrawbeat_localhead_badstress_a050_badcal | report | original_all_10s+ | 0.838633 | 0.855681 | 0.829255 | 0.801186 | 0.944182 | 2896 | 1868 | 98 | nan | nan |
| physiofact_detrawbeat_localhead_badstress_a050 | report | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | nan | nan |
| physiofact_detrawbeat_localhead_badstress_a050_badcal | report | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | nan | nan |
| physiofact_detrawbeat_localhead_badstress_a050 | report | bad_outlier_stress | 0.184932 | 0.104046 | 0.000000 | 0.000000 | 0.184932 | 0 | 0 | 43 | nan | nan |
| physiofact_detrawbeat_localhead_badstress_a050_badcal | report | bad_outlier_stress | 0.184932 | 0.104046 | 0.000000 | 0.000000 | 0.184932 | 0 | 0 | 43 | nan | nan |
| physiofact_detrawbeat_v5sparse_badoutlier_a050 | report | synthetic_val | 0.960442 | 0.950154 | 0.980344 | 0.947917 | 0.980469 | 8 | 0 | 5 | 0.6411 | 0.6575 |
| physiofact_detrawbeat_v5sparse_badoutlier_a050 | report | synthetic_test | 0.978985 | 0.969954 | 0.979079 | 0.978084 | 0.983402 | 9 | 0 | 4 | 0.6871 | 0.6347 |
| physiofact_detrawbeat_v5sparse_badoutlier_a050_badcal | report | synthetic_test | 0.978985 | 0.969954 | 0.979079 | 0.978084 | 0.983402 | 9 | 0 | 4 | nan | nan |
| physiofact_detrawbeat_v5sparse_badoutlier_a050 | report | original_test_all_10s+ | 0.766781 | 0.649342 | 0.874176 | 0.709670 | 0.430657 | 426 | 937 | 65 | nan | nan |
| physiofact_detrawbeat_v5sparse_badoutlier_a050_badcal | report | original_test_all_10s+ | 0.766545 | 0.648989 | 0.874176 | 0.709218 | 0.430657 | 426 | 937 | 65 | nan | nan |
| physiofact_detrawbeat_v5sparse_badoutlier_a050 | report | original_all_10s+ | 0.779099 | 0.805572 | 0.684093 | 0.848702 | 0.945506 | 5276 | 1122 | 118 | nan | nan |
| physiofact_detrawbeat_v5sparse_badoutlier_a050_badcal | report | original_all_10s+ | 0.778948 | 0.805346 | 0.684093 | 0.848231 | 0.945506 | 5276 | 1122 | 118 | nan | nan |
| physiofact_detrawbeat_v5sparse_badoutlier_a050 | report | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | nan | nan |
| physiofact_detrawbeat_v5sparse_badoutlier_a050_badcal | report | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | nan | nan |
| physiofact_detrawbeat_v5sparse_badoutlier_a050 | report | bad_outlier_stress | 0.198630 | 0.110476 | 0.000000 | 0.000000 | 0.198630 | 0 | 0 | 65 | nan | nan |
| physiofact_detrawbeat_v5sparse_badoutlier_a050_badcal | report | bad_outlier_stress | 0.198630 | 0.110476 | 0.000000 | 0.000000 | 0.198630 | 0 | 0 | 65 | nan | nan |
| physiofact_detrawbeat_v5sparse_mediumveto_a050 | report | synthetic_val | 0.912740 | 0.920614 | 1.000000 | 0.863636 | 0.976562 | 0 | 144 | 6 | 0.6418 | 0.6671 |
| physiofact_detrawbeat_v5sparse_mediumveto_a050 | report | synthetic_test | 0.900564 | 0.911467 | 0.993724 | 0.849026 | 0.979253 | 2 | 184 | 5 | 0.6938 | 0.6357 |
| physiofact_detrawbeat_v5sparse_mediumveto_a050_badcal | report | synthetic_test | 0.898514 | 0.908124 | 0.993724 | 0.845779 | 0.979253 | 2 | 184 | 5 | nan | nan |
| physiofact_detrawbeat_v5sparse_mediumveto_a050 | report | original_test_all_10s+ | 0.693760 | 0.604313 | 0.953571 | 0.511523 | 0.355231 | 156 | 2026 | 24 | nan | nan |
| physiofact_detrawbeat_v5sparse_mediumveto_a050_badcal | report | original_test_all_10s+ | 0.693406 | 0.601212 | 0.952747 | 0.506778 | 0.406326 | 156 | 1943 | 22 | nan | nan |
| physiofact_detrawbeat_v5sparse_mediumveto_a050 | report | original_all_10s+ | 0.820761 | 0.823880 | 0.962213 | 0.535661 | 0.937938 | 612 | 4769 | 82 | nan | nan |
| physiofact_detrawbeat_v5sparse_mediumveto_a050_badcal | report | original_all_10s+ | 0.818940 | 0.818542 | 0.960042 | 0.531050 | 0.942857 | 612 | 4670 | 77 | nan | nan |
| physiofact_detrawbeat_v5sparse_mediumveto_a050 | report | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | nan | nan |
| physiofact_detrawbeat_v5sparse_mediumveto_a050_badcal | report | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | nan | nan |
| physiofact_detrawbeat_v5sparse_mediumveto_a050 | report | bad_outlier_stress | 0.092466 | 0.056426 | 0.000000 | 0.000000 | 0.092466 | 0 | 0 | 24 | nan | nan |
| physiofact_detrawbeat_v5sparse_mediumveto_a050_badcal | report | bad_outlier_stress | 0.164384 | 0.094118 | 0.000000 | 0.000000 | 0.164384 | 0 | 0 | 22 | nan | nan |
| physiofact_detrawbeat_v5residual_guard_a050 | report | synthetic_val | 0.968586 | 0.960950 | 0.987715 | 0.959280 | 0.976562 | 5 | 9 | 6 | 0.6701 | 0.6659 |
| physiofact_detrawbeat_v5residual_guard_a050 | report | synthetic_test | 0.989749 | 0.986335 | 0.983264 | 0.994318 | 0.979253 | 7 | 2 | 5 | 0.6945 | 0.6226 |
| physiofact_detrawbeat_v5residual_guard_a050_badcal | report | synthetic_test | 0.990261 | 0.987173 | 0.983264 | 0.994318 | 0.983402 | 7 | 2 | 4 | nan | nan |
| physiofact_detrawbeat_v5residual_guard_a050 | report | original_test_all_10s+ | 0.781409 | 0.676678 | 0.879121 | 0.737912 | 0.384428 | 434 | 1011 | 59 | nan | nan |
| physiofact_detrawbeat_v5residual_guard_a050_badcal | report | original_test_all_10s+ | 0.781644 | 0.679438 | 0.879121 | 0.736557 | 0.403893 | 434 | 1001 | 53 | nan | nan |
| physiofact_detrawbeat_v5residual_guard_a050 | report | original_all_10s+ | 0.812295 | 0.838637 | 0.747228 | 0.852653 | 0.940965 | 4298 | 1384 | 116 | nan | nan |
| physiofact_detrawbeat_v5residual_guard_a050_badcal | report | original_all_10s+ | 0.812265 | 0.838339 | 0.747228 | 0.851807 | 0.942479 | 4298 | 1374 | 110 | nan | nan |
| physiofact_detrawbeat_v5residual_guard_a050 | report | bad_core_nearboundary | 0.991597 | 0.331927 | 0.000000 | 0.000000 | 0.991597 | 0 | 0 | 1 | nan | nan |
| physiofact_detrawbeat_v5residual_guard_a050_badcal | report | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | nan | nan |
| physiofact_detrawbeat_v5residual_guard_a050 | report | bad_outlier_stress | 0.136986 | 0.080321 | 0.000000 | 0.000000 | 0.136986 | 0 | 0 | 58 | nan | nan |
| physiofact_detrawbeat_v5residual_guard_a050_badcal | report | bad_outlier_stress | 0.160959 | 0.092429 | 0.000000 | 0.000000 | 0.160959 | 0 | 0 | 53 | nan | nan |
| physiofact_detrawbeat_v5residual_recall_a050 | report | synthetic_val | 0.968586 | 0.960961 | 0.992629 | 0.957386 | 0.976562 | 3 | 11 | 6 | 0.6704 | 0.6677 |
| physiofact_detrawbeat_v5residual_recall_a050 | report | synthetic_test | 0.990261 | 0.987184 | 0.987448 | 0.992695 | 0.983402 | 5 | 4 | 4 | 0.6949 | 0.6235 |
| physiofact_detrawbeat_v5residual_recall_a050_badcal | report | synthetic_test | 0.990261 | 0.987184 | 0.987448 | 0.992695 | 0.983402 | 5 | 4 | 4 | nan | nan |
| physiofact_detrawbeat_v5residual_recall_a050 | report | original_test_all_10s+ | 0.775746 | 0.672403 | 0.886538 | 0.720515 | 0.389294 | 407 | 1078 | 53 | nan | nan |
| physiofact_detrawbeat_v5residual_recall_a050_badcal | report | original_test_all_10s+ | 0.775628 | 0.672650 | 0.886538 | 0.719837 | 0.394161 | 407 | 1075 | 52 | nan | nan |
| physiofact_detrawbeat_v5residual_recall_a050 | report | original_all_10s+ | 0.824979 | 0.847432 | 0.783313 | 0.833929 | 0.941343 | 3683 | 1571 | 110 | nan | nan |
| physiofact_detrawbeat_v5residual_recall_a050_badcal | report | original_all_10s+ | 0.824948 | 0.847302 | 0.783313 | 0.833459 | 0.942100 | 3683 | 1568 | 107 | nan | nan |
| physiofact_detrawbeat_v5residual_recall_a050 | report | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | nan | nan |
| physiofact_detrawbeat_v5residual_recall_a050_badcal | report | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | nan | nan |
| physiofact_detrawbeat_v5residual_recall_a050 | report | bad_outlier_stress | 0.140411 | 0.082082 | 0.000000 | 0.000000 | 0.140411 | 0 | 0 | 53 | nan | nan |
| physiofact_detrawbeat_v5residual_recall_a050_badcal | report | bad_outlier_stress | 0.147260 | 0.085572 | 0.000000 | 0.000000 | 0.147260 | 0 | 0 | 52 | nan | nan |
| featurefirst_wavecomp_record111physio_guard_a050 | report | synthetic_val | 0.986620 | 0.983797 | 0.987715 | 0.988636 | 0.976562 | 2 | 11 | 6 | 0.7438 | 0.7739 |
| featurefirst_wavecomp_record111physio_guard_a050 | report | synthetic_test | 0.992824 | 0.990587 | 0.993724 | 0.995130 | 0.979253 | 2 | 5 | 5 | 0.7389 | 0.6744 |
| featurefirst_wavecomp_record111physio_guard_a050_badcal | report | synthetic_test | 0.992824 | 0.990587 | 0.993724 | 0.995130 | 0.979253 | 2 | 5 | 5 | nan | nan |
| featurefirst_wavecomp_record111physio_guard_a050 | report | original_test_all_10s+ | 0.860092 | 0.785446 | 0.863462 | 0.877090 | 0.647202 | 492 | 353 | 66 | nan | nan |
| featurefirst_wavecomp_record111physio_guard_a050_badcal | report | original_test_all_10s+ | 0.860210 | 0.786009 | 0.863462 | 0.877090 | 0.649635 | 492 | 353 | 65 | nan | nan |
| featurefirst_wavecomp_record111physio_guard_a050 | report | original_all_10s+ | 0.869220 | 0.884871 | 0.826439 | 0.892642 | 0.960076 | 2952 | 940 | 131 | nan | nan |
| featurefirst_wavecomp_record111physio_guard_a050_badcal | report | original_all_10s+ | 0.869311 | 0.885005 | 0.826439 | 0.892642 | 0.960643 | 2952 | 940 | 128 | nan | nan |
| featurefirst_wavecomp_record111physio_guard_a050 | report | bad_core_nearboundary | 0.907563 | 0.317181 | 0.000000 | 0.000000 | 0.907563 | 0 | 0 | 11 | nan | nan |
| featurefirst_wavecomp_record111physio_guard_a050_badcal | report | bad_core_nearboundary | 0.915966 | 0.318713 | 0.000000 | 0.000000 | 0.915966 | 0 | 0 | 10 | nan | nan |
| featurefirst_wavecomp_record111physio_guard_a050 | report | bad_outlier_stress | 0.541096 | 0.234074 | 0.000000 | 0.000000 | 0.541096 | 0 | 0 | 55 | nan | nan |
| featurefirst_wavecomp_record111physio_guard_a050_badcal | report | bad_outlier_stress | 0.541096 | 0.234074 | 0.000000 | 0.000000 | 0.541096 | 0 | 0 | 55 | nan | nan |
| featurefirst_wavecomp_record111physio_slowlr_warm_a050 | report | synthetic_val | 0.978476 | 0.973406 | 0.970516 | 0.981061 | 0.980469 | 9 | 7 | 5 | 0.7319 | 0.7435 |
| featurefirst_wavecomp_record111physio_slowlr_warm_a050 | report | synthetic_test | 0.990774 | 0.988613 | 0.976987 | 0.998377 | 0.979253 | 10 | 1 | 5 | 0.7307 | 0.6605 |
| featurefirst_wavecomp_record111physio_slowlr_warm_a050_badcal | report | synthetic_test | 0.990774 | 0.988613 | 0.976987 | 0.998377 | 0.979253 | 10 | 1 | 5 | nan | nan |
| featurefirst_wavecomp_record111physio_slowlr_warm_a050 | report | original_test_all_10s+ | 0.842751 | 0.776509 | 0.757418 | 0.931089 | 0.647202 | 881 | 134 | 103 | nan | nan |
| featurefirst_wavecomp_record111physio_slowlr_warm_a050_badcal | report | original_test_all_10s+ | 0.842751 | 0.776509 | 0.757418 | 0.931089 | 0.647202 | 881 | 134 | 103 | nan | nan |
| featurefirst_wavecomp_record111physio_slowlr_warm_a050 | report | original_all_10s+ | 0.848343 | 0.870502 | 0.755970 | 0.941193 | 0.959508 | 4157 | 450 | 171 | nan | nan |
| featurefirst_wavecomp_record111physio_slowlr_warm_a050_badcal | report | original_all_10s+ | 0.848343 | 0.870502 | 0.755970 | 0.941193 | 0.959508 | 4157 | 450 | 171 | nan | nan |
| featurefirst_wavecomp_record111physio_slowlr_warm_a050 | report | bad_core_nearboundary | 0.857143 | 0.307692 | 0.000000 | 0.000000 | 0.857143 | 0 | 0 | 17 | nan | nan |
| featurefirst_wavecomp_record111physio_slowlr_warm_a050_badcal | report | bad_core_nearboundary | 0.857143 | 0.307692 | 0.000000 | 0.000000 | 0.857143 | 0 | 0 | 17 | nan | nan |
| featurefirst_wavecomp_record111physio_slowlr_warm_a050 | report | bad_outlier_stress | 0.561644 | 0.239766 | 0.000000 | 0.000000 | 0.561644 | 0 | 0 | 86 | nan | nan |
| featurefirst_wavecomp_record111physio_slowlr_warm_a050_badcal | report | bad_outlier_stress | 0.561644 | 0.239766 | 0.000000 | 0.000000 | 0.561644 | 0 | 0 | 86 | nan | nan |
| featurefirst_top20_qrsbase_artbad_dualcoreout_recall_a050 | report | synthetic_val | 0.983130 | 0.978154 | 0.987715 | 0.982955 | 0.976562 | 2 | 7 | 6 | 0.6511 | 0.6492 |
| featurefirst_top20_qrsbase_artbad_dualcoreout_recall_a050 | report | synthetic_test | 0.994874 | 0.992516 | 0.993724 | 0.998377 | 0.979253 | 2 | 1 | 5 | 0.6854 | 0.6113 |
| featurefirst_top20_qrsbase_artbad_dualcoreout_recall_a050_badcal | report | synthetic_test | 0.994874 | 0.992516 | 0.993724 | 0.998377 | 0.979253 | 2 | 1 | 5 | nan | nan |
| featurefirst_top20_qrsbase_artbad_dualcoreout_recall_a050 | report | original_test_all_10s+ | 0.863749 | 0.787070 | 0.889560 | 0.864437 | 0.627737 | 397 | 424 | 48 | nan | nan |
| featurefirst_top20_qrsbase_artbad_dualcoreout_recall_a050_badcal | report | original_test_all_10s+ | 0.864103 | 0.788821 | 0.889560 | 0.864437 | 0.635036 | 397 | 424 | 48 | nan | nan |
| featurefirst_top20_qrsbase_artbad_dualcoreout_recall_a050 | report | original_all_10s+ | 0.867338 | 0.883829 | 0.818401 | 0.900263 | 0.958940 | 3088 | 872 | 111 | nan | nan |
| featurefirst_top20_qrsbase_artbad_dualcoreout_recall_a050_badcal | report | original_all_10s+ | 0.867429 | 0.883955 | 0.818401 | 0.900263 | 0.959508 | 3088 | 872 | 111 | nan | nan |
| featurefirst_top20_qrsbase_artbad_dualcoreout_recall_a050 | report | bad_core_nearboundary | 0.966387 | 0.327635 | 0.000000 | 0.000000 | 0.966387 | 0 | 0 | 4 | nan | nan |
| featurefirst_top20_qrsbase_artbad_dualcoreout_recall_a050_badcal | report | bad_core_nearboundary | 0.966387 | 0.327635 | 0.000000 | 0.000000 | 0.966387 | 0 | 0 | 4 | nan | nan |
| featurefirst_top20_qrsbase_artbad_dualcoreout_recall_a050 | report | bad_outlier_stress | 0.489726 | 0.219157 | 0.000000 | 0.000000 | 0.489726 | 0 | 0 | 44 | nan | nan |
| featurefirst_top20_qrsbase_artbad_dualcoreout_recall_a050_badcal | report | bad_outlier_stress | 0.500000 | 0.222222 | 0.000000 | 0.000000 | 0.500000 | 0 | 0 | 44 | nan | nan |
| featurefirst_top20_qrsbase_dualcoreout_gmres_mediumguard_a050 | report | synthetic_val | 0.991274 | 0.990018 | 0.995086 | 0.993371 | 0.976562 | 2 | 7 | 6 | 0.6511 | 0.6492 |
| featurefirst_top20_qrsbase_dualcoreout_gmres_mediumguard_a050 | report | synthetic_test | 0.995900 | 0.994379 | 0.995816 | 0.999188 | 0.979253 | 2 | 1 | 5 | 0.6854 | 0.6113 |
| featurefirst_top20_qrsbase_dualcoreout_gmres_mediumguard_a050_badcal | report | synthetic_test | 0.995387 | 0.993339 | 0.993724 | 0.999188 | 0.979253 | 2 | 1 | 5 | nan | nan |
| featurefirst_top20_qrsbase_dualcoreout_gmres_mediumguard_a050 | report | original_test_all_10s+ | 0.855019 | 0.733692 | 0.894505 | 0.869182 | 0.352798 | 381 | 511 | 80 | nan | nan |
| featurefirst_top20_qrsbase_dualcoreout_gmres_mediumguard_a050_badcal | report | original_test_all_10s+ | 0.857497 | 0.758897 | 0.894505 | 0.862178 | 0.479319 | 380 | 483 | 61 | nan | nan |
| featurefirst_top20_qrsbase_dualcoreout_gmres_mediumguard_a050 | report | original_all_10s+ | 0.866519 | 0.883518 | 0.822742 | 0.901487 | 0.937370 | 3017 | 973 | 144 | nan | nan |
| featurefirst_top20_qrsbase_dualcoreout_gmres_mediumguard_a050_badcal | report | original_all_10s+ | 0.867065 | 0.883523 | 0.822742 | 0.898288 | 0.947209 | 3016 | 945 | 125 | nan | nan |
| featurefirst_top20_qrsbase_dualcoreout_gmres_mediumguard_a050 | report | bad_core_nearboundary | 0.957983 | 0.326180 | 0.000000 | 0.000000 | 0.957983 | 0 | 0 | 5 | nan | nan |
| featurefirst_top20_qrsbase_dualcoreout_gmres_mediumguard_a050_badcal | report | bad_core_nearboundary | 0.983193 | 0.330508 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 2 | nan | nan |
| featurefirst_top20_qrsbase_dualcoreout_gmres_mediumguard_a050 | report | bad_outlier_stress | 0.106164 | 0.063983 | 0.000000 | 0.000000 | 0.106164 | 0 | 0 | 75 | nan | nan |
| featurefirst_top20_qrsbase_dualcoreout_gmres_mediumguard_a050_badcal | report | bad_outlier_stress | 0.273973 | 0.143369 | 0.000000 | 0.000000 | 0.273973 | 0 | 0 | 59 | nan | nan |
| featurefirst_top20_qrsbase_dualcoreout_v5badbranch_preserveartifact_balanced_a050 | report | synthetic_val | 0.990692 | 0.989210 | 0.995086 | 0.992424 | 0.976562 | 2 | 7 | 6 | 0.6511 | 0.6492 |
| featurefirst_top20_qrsbase_dualcoreout_v5badbranch_preserveartifact_balanced_a050 | report | synthetic_test | 0.995387 | 0.993339 | 0.993724 | 0.999188 | 0.979253 | 2 | 1 | 5 | 0.6854 | 0.6113 |
| featurefirst_top20_qrsbase_dualcoreout_v5badbranch_preserveartifact_balanced_a050_badcal | report | synthetic_test | 0.995387 | 0.993339 | 0.993724 | 0.999188 | 0.979253 | 2 | 1 | 5 | nan | nan |
| featurefirst_top20_qrsbase_dualcoreout_v5badbranch_preserveartifact_balanced_a050 | report | original_test_all_10s+ | 0.860918 | 0.770934 | 0.889560 | 0.865793 | 0.554745 | 395 | 428 | 54 | nan | nan |
| featurefirst_top20_qrsbase_dualcoreout_v5badbranch_preserveartifact_balanced_a050_badcal | report | original_test_all_10s+ | 0.862451 | 0.778927 | 0.889560 | 0.865567 | 0.588808 | 395 | 426 | 52 | nan | nan |
| featurefirst_top20_qrsbase_dualcoreout_v5badbranch_preserveartifact_balanced_a050 | report | original_all_10s+ | 0.866640 | 0.882934 | 0.818401 | 0.901016 | 0.953075 | 3086 | 876 | 118 | nan | nan |
| featurefirst_top20_qrsbase_dualcoreout_v5badbranch_preserveartifact_balanced_a050_badcal | report | original_all_10s+ | 0.867035 | 0.883439 | 0.818401 | 0.900922 | 0.955724 | 3086 | 874 | 116 | nan | nan |
| featurefirst_top20_qrsbase_dualcoreout_v5badbranch_preserveartifact_balanced_a050 | report | bad_core_nearboundary | 0.949580 | 0.324713 | 0.000000 | 0.000000 | 0.949580 | 0 | 0 | 6 | nan | nan |
| featurefirst_top20_qrsbase_dualcoreout_v5badbranch_preserveartifact_balanced_a050_badcal | report | bad_core_nearboundary | 0.949580 | 0.324713 | 0.000000 | 0.000000 | 0.949580 | 0 | 0 | 6 | nan | nan |
| featurefirst_top20_qrsbase_dualcoreout_v5badbranch_preserveartifact_balanced_a050 | report | bad_outlier_stress | 0.393836 | 0.188370 | 0.000000 | 0.000000 | 0.393836 | 0 | 0 | 48 | nan | nan |
| featurefirst_top20_qrsbase_dualcoreout_v5badbranch_preserveartifact_balanced_a050_badcal | report | bad_outlier_stress | 0.441781 | 0.204276 | 0.000000 | 0.000000 | 0.441781 | 0 | 0 | 46 | nan | nan |
| featurefirst_top20_qrsbase_dualcoreout_badfeat_current_balanced_a050 | report | synthetic_val | 0.991274 | 0.990018 | 0.995086 | 0.993371 | 0.976562 | 2 | 7 | 6 | 0.6511 | 0.6492 |
| featurefirst_top20_qrsbase_dualcoreout_badfeat_current_balanced_a050 | report | synthetic_test | 0.995900 | 0.994379 | 0.995816 | 0.999188 | 0.979253 | 2 | 1 | 5 | 0.6854 | 0.6113 |
| featurefirst_top20_qrsbase_dualcoreout_badfeat_current_balanced_a050_badcal | report | synthetic_test | 0.995387 | 0.993339 | 0.993724 | 0.999188 | 0.979253 | 2 | 1 | 5 | nan | nan |
| featurefirst_top20_qrsbase_dualcoreout_badfeat_current_balanced_a050 | report | original_test_all_10s+ | 0.857379 | 0.729270 | 0.889835 | 0.880479 | 0.321168 | 399 | 481 | 91 | nan | nan |
| featurefirst_top20_qrsbase_dualcoreout_badfeat_current_balanced_a050_badcal | report | original_test_all_10s+ | 0.861508 | 0.769764 | 0.889835 | 0.869634 | 0.523114 | 395 | 439 | 60 | nan | nan |
| featurefirst_top20_qrsbase_dualcoreout_badfeat_current_balanced_a050 | report | original_all_10s+ | 0.865912 | 0.883435 | 0.818459 | 0.907697 | 0.934910 | 3091 | 929 | 155 | nan | nan |
| featurefirst_top20_qrsbase_dualcoreout_badfeat_current_balanced_a050_badcal | report | original_all_10s+ | 0.866883 | 0.883590 | 0.818459 | 0.902710 | 0.950993 | 3086 | 887 | 122 | nan | nan |
| featurefirst_top20_qrsbase_dualcoreout_badfeat_current_balanced_a050 | report | bad_core_nearboundary | 0.949580 | 0.324713 | 0.000000 | 0.000000 | 0.949580 | 0 | 0 | 6 | nan | nan |
| featurefirst_top20_qrsbase_dualcoreout_badfeat_current_balanced_a050_badcal | report | bad_core_nearboundary | 0.966387 | 0.327635 | 0.000000 | 0.000000 | 0.966387 | 0 | 0 | 4 | nan | nan |
| featurefirst_top20_qrsbase_dualcoreout_badfeat_current_balanced_a050 | report | bad_outlier_stress | 0.065068 | 0.040729 | 0.000000 | 0.000000 | 0.065068 | 0 | 0 | 85 | nan | nan |
| featurefirst_top20_qrsbase_dualcoreout_badfeat_current_balanced_a050_badcal | report | bad_outlier_stress | 0.342466 | 0.170068 | 0.000000 | 0.000000 | 0.342466 | 0 | 0 | 56 | nan | nan |
| featurefirst_top20_qrsbase_primres_current_balanced_a050 | report | synthetic_val | 0.983130 | 0.979793 | 0.995086 | 0.980114 | 0.976562 | 2 | 11 | 6 | 0.6519 | 0.6537 |
| featurefirst_top20_qrsbase_primres_current_balanced_a050 | report | synthetic_test | 0.993337 | 0.991410 | 0.993724 | 0.995942 | 0.979253 | 2 | 5 | 5 | 0.6872 | 0.6151 |
| featurefirst_top20_qrsbase_primres_current_balanced_a050_badcal | report | synthetic_test | 0.993337 | 0.991410 | 0.993724 | 0.995942 | 0.979253 | 2 | 5 | 5 | nan | nan |
| featurefirst_top20_qrsbase_primres_current_balanced_a050 | report | original_test_all_10s+ | 0.825764 | 0.734301 | 0.908516 | 0.789200 | 0.486618 | 327 | 785 | 45 | nan | nan |
| featurefirst_top20_qrsbase_primres_current_balanced_a050_badcal | report | original_test_all_10s+ | 0.825764 | 0.734301 | 0.908516 | 0.789200 | 0.486618 | 327 | 785 | 45 | nan | nan |
| featurefirst_top20_qrsbase_primres_current_balanced_a050 | report | original_all_10s+ | 0.866883 | 0.881432 | 0.849968 | 0.853688 | 0.947966 | 2547 | 1389 | 108 | nan | nan |
| featurefirst_top20_qrsbase_primres_current_balanced_a050_badcal | report | original_all_10s+ | 0.866883 | 0.881432 | 0.849968 | 0.853688 | 0.947966 | 2547 | 1389 | 108 | nan | nan |
| featurefirst_top20_qrsbase_primres_current_balanced_a050 | report | bad_core_nearboundary | 0.957983 | 0.326180 | 0.000000 | 0.000000 | 0.957983 | 0 | 0 | 5 | nan | nan |
| featurefirst_top20_qrsbase_primres_current_balanced_a050_badcal | report | bad_core_nearboundary | 0.957983 | 0.326180 | 0.000000 | 0.000000 | 0.957983 | 0 | 0 | 5 | nan | nan |
| featurefirst_top20_qrsbase_primres_current_balanced_a050 | report | bad_outlier_stress | 0.294521 | 0.151675 | 0.000000 | 0.000000 | 0.294521 | 0 | 0 | 40 | nan | nan |
| featurefirst_top20_qrsbase_primres_current_balanced_a050_badcal | report | bad_outlier_stress | 0.294521 | 0.151675 | 0.000000 | 0.000000 | 0.294521 | 0 | 0 | 40 | nan | nan |
| featurefirst_top20_qrsbase_primauxres2_current_calibrated_a050 | report | synthetic_val | 0.988947 | 0.987799 | 1.000000 | 0.987689 | 0.976562 | 0 | 13 | 6 | 0.6513 | 0.6502 |
| featurefirst_top20_qrsbase_primauxres2_current_calibrated_a050 | report | synthetic_test | 0.991799 | 0.990532 | 0.995816 | 0.992695 | 0.979253 | 2 | 9 | 5 | 0.6868 | 0.6141 |
| featurefirst_top20_qrsbase_primauxres2_current_calibrated_a050_badcal | report | synthetic_test | 0.991799 | 0.990532 | 0.995816 | 0.992695 | 0.979253 | 2 | 9 | 5 | nan | nan |
| featurefirst_top20_qrsbase_primauxres2_current_calibrated_a050 | report | original_test_all_10s+ | 0.820573 | 0.701142 | 0.913462 | 0.792363 | 0.301703 | 312 | 888 | 69 | nan | nan |
| featurefirst_top20_qrsbase_primauxres2_current_calibrated_a050_badcal | report | original_test_all_10s+ | 0.822107 | 0.720080 | 0.913462 | 0.786941 | 0.391727 | 311 | 861 | 50 | nan | nan |
| featurefirst_top20_qrsbase_primauxres2_current_calibrated_a050 | report | original_all_10s+ | 0.870555 | 0.885069 | 0.872910 | 0.835623 | 0.933207 | 2162 | 1710 | 133 | nan | nan |
| featurefirst_top20_qrsbase_primauxres2_current_calibrated_a050_badcal | report | original_all_10s+ | 0.870919 | 0.884816 | 0.872910 | 0.832894 | 0.940965 | 2160 | 1683 | 111 | nan | nan |
| featurefirst_top20_qrsbase_primauxres2_current_calibrated_a050 | report | bad_core_nearboundary | 0.974790 | 0.329078 | 0.000000 | 0.000000 | 0.974790 | 0 | 0 | 3 | nan | nan |
| featurefirst_top20_qrsbase_primauxres2_current_calibrated_a050_badcal | report | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | nan | nan |
| featurefirst_top20_qrsbase_primauxres2_current_calibrated_a050 | report | bad_outlier_stress | 0.027397 | 0.017778 | 0.000000 | 0.000000 | 0.027397 | 0 | 0 | 66 | nan | nan |
| featurefirst_top20_qrsbase_primauxres2_current_calibrated_a050_badcal | report | bad_outlier_stress | 0.143836 | 0.083832 | 0.000000 | 0.000000 | 0.143836 | 0 | 0 | 50 | nan | nan |
| featurefirst_top20_qrsbase_v5bank_flatlowamp_auxlite_a050 | report | synthetic_val | 0.984875 | 0.982819 | 1.000000 | 0.984848 | 0.960938 | 0 | 16 | 10 | 0.6530 | 0.6570 |
| featurefirst_top20_qrsbase_v5bank_flatlowamp_auxlite_a050 | report | synthetic_test | 0.987699 | 0.986355 | 0.995816 | 0.987013 | 0.975104 | 2 | 16 | 6 | 0.6909 | 0.6215 |
| featurefirst_top20_qrsbase_v5bank_flatlowamp_auxlite_a050_badcal | report | synthetic_test | 0.987186 | 0.985333 | 0.993724 | 0.986201 | 0.979253 | 2 | 16 | 5 | nan | nan |
| featurefirst_top20_qrsbase_v5bank_flatlowamp_auxlite_a050 | report | original_test_all_10s+ | 0.750855 | 0.528002 | 0.928022 | 0.672616 | 0.024331 | 261 | 1431 | 146 | nan | nan |
| featurefirst_top20_qrsbase_v5bank_flatlowamp_auxlite_a050_badcal | report | original_test_all_10s+ | 0.767135 | 0.684301 | 0.927747 | 0.669453 | 0.396594 | 258 | 1387 | 37 | nan | nan |
| featurefirst_top20_qrsbase_v5bank_flatlowamp_auxlite_a050 | report | original_all_10s+ | 0.825859 | 0.821329 | 0.886346 | 0.773805 | 0.735478 | 1936 | 2382 | 1140 | nan | nan |
| featurefirst_top20_qrsbase_v5bank_flatlowamp_auxlite_a050_badcal | report | original_all_10s+ | 0.858296 | 0.872839 | 0.886288 | 0.772111 | 0.941343 | 1931 | 2337 | 97 | nan | nan |
| featurefirst_top20_qrsbase_v5bank_flatlowamp_auxlite_a050 | report | bad_core_nearboundary | 0.042017 | 0.026882 | 0.000000 | 0.000000 | 0.042017 | 0 | 0 | 114 | nan | nan |
| featurefirst_top20_qrsbase_v5bank_flatlowamp_auxlite_a050_badcal | report | bad_core_nearboundary | 0.924370 | 0.320233 | 0.000000 | 0.000000 | 0.924370 | 0 | 0 | 9 | nan | nan |
| featurefirst_top20_qrsbase_v5bank_flatlowamp_auxlite_a050 | report | bad_outlier_stress | 0.017123 | 0.011223 | 0.000000 | 0.000000 | 0.017123 | 0 | 0 | 32 | nan | nan |
| featurefirst_top20_qrsbase_v5bank_flatlowamp_auxlite_a050_badcal | report | bad_outlier_stress | 0.181507 | 0.102415 | 0.000000 | 0.000000 | 0.181507 | 0 | 0 | 28 | nan | nan |
| featurefirst_top20_qrsbase_flatlowamp_balanced_a050 | report | synthetic_val | 0.991274 | 0.990018 | 0.995086 | 0.993371 | 0.976562 | 2 | 7 | 6 | 0.6511 | 0.6492 |
| featurefirst_top20_qrsbase_flatlowamp_balanced_a050 | report | synthetic_test | 0.995900 | 0.994379 | 0.995816 | 0.999188 | 0.979253 | 2 | 1 | 5 | 0.6854 | 0.6113 |
| featurefirst_top20_qrsbase_flatlowamp_balanced_a050_badcal | report | synthetic_test | 0.995387 | 0.993339 | 0.993724 | 0.999188 | 0.979253 | 2 | 1 | 5 | nan | nan |
| featurefirst_top20_qrsbase_flatlowamp_balanced_a050 | report | original_test_all_10s+ | 0.856081 | 0.700427 | 0.889835 | 0.886353 | 0.231144 | 400 | 481 | 129 | nan | nan |
| featurefirst_top20_qrsbase_flatlowamp_balanced_a050_badcal | report | original_test_all_10s+ | 0.859856 | 0.756826 | 0.889835 | 0.872797 | 0.454988 | 395 | 451 | 69 | nan | nan |
| featurefirst_top20_qrsbase_flatlowamp_balanced_a050 | report | original_all_10s+ | 0.865578 | 0.883048 | 0.818459 | 0.910519 | 0.927152 | 3093 | 929 | 197 | nan | nan |
| featurefirst_top20_qrsbase_flatlowamp_balanced_a050_badcal | report | original_all_10s+ | 0.866458 | 0.883220 | 0.818459 | 0.903651 | 0.946452 | 3082 | 899 | 127 | nan | nan |
| featurefirst_top20_qrsbase_flatlowamp_balanced_a050 | report | bad_core_nearboundary | 0.647059 | 0.261905 | 0.000000 | 0.000000 | 0.647059 | 0 | 0 | 42 | nan | nan |
| featurefirst_top20_qrsbase_flatlowamp_balanced_a050_badcal | report | bad_core_nearboundary | 1.000000 | 0.333333 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 | nan | nan |
| featurefirst_top20_qrsbase_flatlowamp_balanced_a050 | report | bad_outlier_stress | 0.061644 | 0.038710 | 0.000000 | 0.000000 | 0.061644 | 0 | 0 | 87 | nan | nan |
| featurefirst_top20_qrsbase_flatlowamp_balanced_a050_badcal | report | bad_outlier_stress | 0.232877 | 0.125926 | 0.000000 | 0.000000 | 0.232877 | 0 | 0 | 69 | nan | nan |

## Baselines

- `47_feature_tabular_upper_bound` original_test_all_10s+: acc `0.963548`, recalls 0.956/0.973/0.927`. tabular-only upper bound; geometry features are inference inputs.
- `current_best_waveform_transformer` original_test_all_10s+: acc `0.811018`, recalls 0.961/0.726/0.401`. aug_convtx_balanced_focal raw.
- `stattoken_baseline` original_test_all_10s+: acc `0.813141`, recalls 0.962/0.726/0.433`. stattx_medium_guard raw.
- `reduced_feature_auto_topk20` original_test_all_10s+: acc `0.913531`, recalls 0.973/0.885/0.696`. reduced tabular model; not waveform-only.

## Candidates

- `physiofact_detrawbeat_localhead_badstress_a050` (report): best_epoch=2, useful_gate=pass, original_report_ran=True, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_geometry_student\N17043_gm_probe\search\physiofact_detrawbeat_localhead_badstress_a050\ckpt_best.pt`
- `physiofact_detrawbeat_v5sparse_badoutlier_a050` (report): best_epoch=2, useful_gate=pass, original_report_ran=True, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_geometry_student\N17043_gm_probe\search\physiofact_detrawbeat_v5sparse_badoutlier_a050\ckpt_best.pt`
- `physiofact_detrawbeat_v5sparse_mediumveto_a050` (report): best_epoch=2, useful_gate=fail, original_report_ran=True, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_geometry_student\N17043_gm_probe\search\physiofact_detrawbeat_v5sparse_mediumveto_a050\ckpt_best.pt`
- `physiofact_detrawbeat_v5residual_guard_a050` (report): best_epoch=5, useful_gate=pass, original_report_ran=True, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_geometry_student\N17043_gm_probe\search\physiofact_detrawbeat_v5residual_guard_a050\ckpt_best.pt`
- `physiofact_detrawbeat_v5residual_recall_a050` (report): best_epoch=2, useful_gate=pass, original_report_ran=True, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_geometry_student\N17043_gm_probe\search\physiofact_detrawbeat_v5residual_recall_a050\ckpt_best.pt`
- `featurefirst_wavecomp_record111physio_guard_a050` (report): best_epoch=2, useful_gate=pass, original_report_ran=True, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_geometry_student\N17043_gm_probe\search\featurefirst_wavecomp_record111physio_guard_a050\ckpt_best.pt`
- `featurefirst_wavecomp_record111physio_slowlr_warm_a050` (report): best_epoch=8, useful_gate=pass, original_report_ran=True, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_geometry_student\N17043_gm_probe\search\featurefirst_wavecomp_record111physio_slowlr_warm_a050\ckpt_best.pt`
- `featurefirst_top20_qrsbase_artbad_dualcoreout_recall_a050` (report): best_epoch=5, useful_gate=pass, original_report_ran=True, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_geometry_student\N17043_gm_probe\search\featurefirst_top20_qrsbase_artbad_dualcoreout_recall_a050\ckpt_best.pt`
- `featurefirst_top20_qrsbase_dualcoreout_gmres_mediumguard_a050` (report): best_epoch=5, useful_gate=pass, original_report_ran=True, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_geometry_student\N17043_gm_probe\search\featurefirst_top20_qrsbase_dualcoreout_gmres_mediumguard_a050\ckpt_best.pt`
- `featurefirst_top20_qrsbase_dualcoreout_v5badbranch_preserveartifact_balanced_a050` (report): best_epoch=2, useful_gate=pass, original_report_ran=True, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_geometry_student\N17043_gm_probe\search\featurefirst_top20_qrsbase_dualcoreout_v5badbranch_preserveartifact_balanced_a050\ckpt_best.pt`
- `featurefirst_top20_qrsbase_dualcoreout_badfeat_current_balanced_a050` (report): best_epoch=4, useful_gate=pass, original_report_ran=True, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_geometry_student\N17043_gm_probe\search\featurefirst_top20_qrsbase_dualcoreout_badfeat_current_balanced_a050\ckpt_best.pt`
- `featurefirst_top20_qrsbase_primres_current_balanced_a050` (report): best_epoch=4, useful_gate=pass, original_report_ran=True, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_geometry_student\N17043_gm_probe\search\featurefirst_top20_qrsbase_primres_current_balanced_a050\ckpt_best.pt`
- `featurefirst_top20_qrsbase_primauxres2_current_calibrated_a050` (report): best_epoch=8, useful_gate=pass, original_report_ran=True, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_geometry_student\N17043_gm_probe\search\featurefirst_top20_qrsbase_primauxres2_current_calibrated_a050\ckpt_best.pt`
- `featurefirst_top20_qrsbase_v5bank_flatlowamp_auxlite_a050` (report): best_epoch=5, useful_gate=pass, original_report_ran=True, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_geometry_student\N17043_gm_probe\search\featurefirst_top20_qrsbase_v5bank_flatlowamp_auxlite_a050\ckpt_best.pt`
- `featurefirst_top20_qrsbase_flatlowamp_balanced_a050` (report): best_epoch=5, useful_gate=pass, original_report_ran=True, checkpoint=`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\runs\waveform_geometry_student\N17043_gm_probe\search\featurefirst_top20_qrsbase_flatlowamp_balanced_a050\ckpt_best.pt`

## Notes

- Selection uses synthetic/node diagnostic only.
- BUT buckets are emitted only for useful-gate candidates unless `--but-report-mode always` is used.
- Diagnostic artifacts are saved beside each checkpoint: teacher recovery CSV, error rows, embedding PCA, and waveform mistake panels.
- Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\waveform_geometry_student_report_metrics.csv`
