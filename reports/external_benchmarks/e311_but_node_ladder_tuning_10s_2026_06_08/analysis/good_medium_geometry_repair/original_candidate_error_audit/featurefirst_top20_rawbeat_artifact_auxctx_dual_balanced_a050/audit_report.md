# Original Candidate Error Audit: featurefirst_top20_rawbeat_artifact_auxctx_dual_balanced_a050

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.843695 | 0.870879 | 0.868730 | 0.333333 | 461 | 535 | 99 |
| raw | original_all_10s+ | 0.820579 | 0.719709 | 0.924539 | 0.936802 | 4759 | 745 | 157 |
| raw | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| raw | bad_outlier_stress | 0.061644 | 0.000000 | 0.000000 | 0.061644 | 0 | 0 | 99 |
| badcal | original_test_all_10s+ | 0.820219 | 0.869505 | 0.813150 | 0.459854 | 447 | 467 | 67 |
| badcal | original_all_10s+ | 0.811810 | 0.718653 | 0.893583 | 0.947777 | 4573 | 673 | 120 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.239726 | 0.000000 | 0.000000 | 0.239726 | 0 | 0 | 67 |

## Error Counts

- test errors raw: 1325
- bad outlier errors raw: 274
- bad core errors raw: 0
- good->medium raw: 461
- medium->good raw: 535
- nonbad->bad raw: 55

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featurefirst_top20_rawbeat_artifact_auxctx_dual_balanced_a050/original_error_waveform_panels.png)
