# Original Candidate Error Audit: featurefirst_quality20_currentbest_headreset_balanced_a050

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.789194 | 0.923901 | 0.723904 | 0.299270 | 273 | 1191 | 46 |
| raw | original_all_10s+ | 0.863758 | 0.883002 | 0.798457 | 0.933018 | 1989 | 2104 | 109 |
| raw | bad_core_nearboundary | 0.957983 | 0.000000 | 0.000000 | 0.957983 | 0 | 0 | 5 |
| raw | bad_outlier_stress | 0.030822 | 0.000000 | 0.000000 | 0.030822 | 0 | 0 | 41 |
| badcal | original_test_all_10s+ | 0.789666 | 0.923901 | 0.704248 | 0.520681 | 268 | 1095 | 24 |
| badcal | original_all_10s+ | 0.863545 | 0.883002 | 0.788577 | 0.951561 | 1977 | 2008 | 81 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.325342 | 0.000000 | 0.000000 | 0.325342 | 0 | 0 | 24 |

## Error Counts

- test errors raw: 1787
- bad outlier errors raw: 283
- bad core errors raw: 5
- good->medium raw: 273
- medium->good raw: 1191
- nonbad->bad raw: 35

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featurefirst_quality20_currentbest_headreset_balanced_a050/original_error_waveform_panels.png)
