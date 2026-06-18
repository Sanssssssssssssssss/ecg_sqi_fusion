# Original Candidate Error Audit: featurefirst_top20_qrsbase_dualcoreout_badfeat_current_recall_a050

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.857379 | 0.889835 | 0.880253 | 0.323601 | 399 | 481 | 90 |
| raw | original_all_10s+ | 0.865912 | 0.818459 | 0.907603 | 0.935099 | 3092 | 929 | 154 |
| raw | bad_core_nearboundary | 0.949580 | 0.000000 | 0.000000 | 0.949580 | 0 | 0 | 6 |
| raw | bad_outlier_stress | 0.068493 | 0.000000 | 0.000000 | 0.068493 | 0 | 0 | 84 |
| badcal | original_test_all_10s+ | 0.862097 | 0.889835 | 0.868730 | 0.545012 | 395 | 436 | 57 |
| badcal | original_all_10s+ | 0.867004 | 0.818459 | 0.902239 | 0.952696 | 3086 | 884 | 119 |
| badcal | bad_core_nearboundary | 0.983193 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 2 |
| badcal | bad_outlier_stress | 0.366438 | 0.000000 | 0.000000 | 0.366438 | 0 | 0 | 55 |

## Error Counts

- test errors raw: 1209
- bad outlier errors raw: 272
- bad core errors raw: 6
- good->medium raw: 399
- medium->good raw: 481
- nonbad->bad raw: 51

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featurefirst_top20_qrsbase_dualcoreout_badfeat_current_recall_a050/original_error_waveform_panels.png)
