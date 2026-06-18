# Original Candidate Error Audit: featurefirst_top20_qrsbase_flatlowamp_guard_a050

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.857733 | 0.889835 | 0.886579 | 0.262774 | 400 | 479 | 119 |
| raw | original_all_10s+ | 0.866125 | 0.818459 | 0.910613 | 0.930369 | 3093 | 927 | 183 |
| raw | bad_core_nearboundary | 0.714286 | 0.000000 | 0.000000 | 0.714286 | 0 | 0 | 34 |
| raw | bad_outlier_stress | 0.078767 | 0.000000 | 0.000000 | 0.078767 | 0 | 0 | 85 |
| badcal | original_test_all_10s+ | 0.861744 | 0.889835 | 0.881157 | 0.403893 | 399 | 461 | 83 |
| badcal | original_all_10s+ | 0.867035 | 0.818459 | 0.907603 | 0.942100 | 3091 | 909 | 143 |
| badcal | bad_core_nearboundary | 0.941176 | 0.000000 | 0.000000 | 0.941176 | 0 | 0 | 7 |
| badcal | bad_outlier_stress | 0.184932 | 0.000000 | 0.000000 | 0.184932 | 0 | 0 | 76 |

## Error Counts

- test errors raw: 1206
- bad outlier errors raw: 269
- bad core errors raw: 34
- good->medium raw: 400
- medium->good raw: 479
- nonbad->bad raw: 24

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featurefirst_top20_qrsbase_flatlowamp_guard_a050/original_error_waveform_panels.png)
