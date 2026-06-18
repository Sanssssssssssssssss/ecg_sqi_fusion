# Original Candidate Error Audit: featurefirst_top20_shift_stress_a050

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.837325 | 0.834066 | 0.885450 | 0.347932 | 582 | 411 | 80 |
| raw | original_all_10s+ | 0.845764 | 0.773925 | 0.915318 | 0.937559 | 3804 | 744 | 140 |
| raw | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| raw | bad_outlier_stress | 0.082192 | 0.000000 | 0.000000 | 0.082192 | 0 | 0 | 80 |
| badcal | original_test_all_10s+ | 0.829303 | 0.831868 | 0.870538 | 0.362530 | 575 | 411 | 76 |
| badcal | original_all_10s+ | 0.841395 | 0.773221 | 0.902333 | 0.938694 | 3777 | 744 | 136 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.102740 | 0.000000 | 0.000000 | 0.102740 | 0 | 0 | 76 |

## Error Counts

- test errors raw: 1379
- bad outlier errors raw: 268
- bad core errors raw: 0
- good->medium raw: 582
- medium->good raw: 411
- nonbad->bad raw: 118

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featurefirst_top20_shift_stress_a050/original_error_waveform_panels.png)
