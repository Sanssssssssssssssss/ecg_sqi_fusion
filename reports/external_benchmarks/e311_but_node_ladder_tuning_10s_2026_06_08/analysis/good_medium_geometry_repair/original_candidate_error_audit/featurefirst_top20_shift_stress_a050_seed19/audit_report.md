# Original Candidate Error Audit: featurefirst_top20_shift_stress_a050_seed19

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.793087 | 0.905769 | 0.747854 | 0.282238 | 342 | 1104 | 61 |
| raw | original_all_10s+ | 0.850012 | 0.831192 | 0.839575 | 0.931693 | 2875 | 1688 | 124 |
| raw | bad_core_nearboundary | 0.957983 | 0.000000 | 0.000000 | 0.957983 | 0 | 0 | 5 |
| raw | bad_outlier_stress | 0.006849 | 0.000000 | 0.000000 | 0.006849 | 0 | 0 | 56 |
| badcal | original_test_all_10s+ | 0.789902 | 0.903571 | 0.741301 | 0.306569 | 339 | 1061 | 55 |
| badcal | original_all_10s+ | 0.848525 | 0.829842 | 0.835717 | 0.934532 | 2870 | 1645 | 113 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.023973 | 0.000000 | 0.000000 | 0.023973 | 0 | 0 | 55 |

## Error Counts

- test errors raw: 1754
- bad outlier errors raw: 290
- bad core errors raw: 5
- good->medium raw: 342
- medium->good raw: 1104
- nonbad->bad raw: 13

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featurefirst_top20_shift_stress_a050_seed19/original_error_waveform_panels.png)
