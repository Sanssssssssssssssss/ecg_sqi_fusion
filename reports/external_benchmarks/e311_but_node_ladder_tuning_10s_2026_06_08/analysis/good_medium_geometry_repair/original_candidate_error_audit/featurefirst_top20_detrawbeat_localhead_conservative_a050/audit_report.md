# Original Candidate Error Audit: featurefirst_top20_detrawbeat_localhead_conservative_a050

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.760764 | 0.924451 | 0.668549 | 0.304136 | 271 | 1426 | 46 |
| raw | original_all_10s+ | 0.834112 | 0.818224 | 0.809842 | 0.934153 | 3093 | 1973 | 105 |
| raw | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| raw | bad_outlier_stress | 0.020548 | 0.000000 | 0.000000 | 0.020548 | 0 | 0 | 46 |
| badcal | original_test_all_10s+ | 0.758759 | 0.924451 | 0.655219 | 0.406326 | 264 | 1381 | 35 |
| badcal | original_all_10s+ | 0.833202 | 0.818224 | 0.802691 | 0.942857 | 3076 | 1928 | 91 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.164384 | 0.000000 | 0.000000 | 0.164384 | 0 | 0 | 35 |

## Error Counts

- test errors raw: 2028
- bad outlier errors raw: 286
- bad core errors raw: 0
- good->medium raw: 271
- medium->good raw: 1426
- nonbad->bad raw: 45

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featurefirst_top20_detrawbeat_localhead_conservative_a050/original_error_waveform_panels.png)
