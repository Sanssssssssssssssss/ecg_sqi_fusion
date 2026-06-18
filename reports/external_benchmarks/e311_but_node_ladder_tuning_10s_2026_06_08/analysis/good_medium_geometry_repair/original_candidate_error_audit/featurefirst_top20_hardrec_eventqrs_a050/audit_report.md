# Original Candidate Error Audit: featurefirst_top20_hardrec_eventqrs_a050

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.808777 | 0.903022 | 0.778581 | 0.299270 | 351 | 969 | 73 |
| raw | original_all_10s+ | 0.843276 | 0.796573 | 0.873824 | 0.932450 | 3465 | 1325 | 139 |
| raw | bad_core_nearboundary | 0.983193 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 2 |
| raw | bad_outlier_stress | 0.020548 | 0.000000 | 0.000000 | 0.020548 | 0 | 0 | 71 |
| badcal | original_test_all_10s+ | 0.805828 | 0.900549 | 0.772481 | 0.326034 | 351 | 939 | 67 |
| badcal | original_all_10s+ | 0.842396 | 0.795811 | 0.870342 | 0.936424 | 3463 | 1294 | 124 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.051370 | 0.000000 | 0.000000 | 0.051370 | 0 | 0 | 67 |

## Error Counts

- test errors raw: 1621
- bad outlier errors raw: 286
- bad core errors raw: 2
- good->medium raw: 351
- medium->good raw: 969
- nonbad->bad raw: 13

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featurefirst_top20_hardrec_eventqrs_a050/original_error_waveform_panels.png)
