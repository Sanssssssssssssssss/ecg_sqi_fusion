# Original Candidate Error Audit: featuregate_top20_shift_stress_goodguard_a050_bneg

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.818922 | 0.906319 | 0.800045 | 0.248175 | 341 | 881 | 90 |
| raw | original_all_10s+ | 0.861209 | 0.839230 | 0.862721 | 0.929044 | 2740 | 1452 | 154 |
| raw | bad_core_nearboundary | 0.823529 | 0.000000 | 0.000000 | 0.823529 | 0 | 0 | 21 |
| raw | bad_outlier_stress | 0.013699 | 0.000000 | 0.000000 | 0.013699 | 0 | 0 | 69 |
| badcal | original_test_all_10s+ | 0.811490 | 0.904670 | 0.778355 | 0.343066 | 335 | 844 | 58 |
| badcal | original_all_10s+ | 0.858417 | 0.838643 | 0.850772 | 0.937559 | 2727 | 1413 | 116 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.075342 | 0.000000 | 0.000000 | 0.075342 | 0 | 0 | 58 |

## Error Counts

- test errors raw: 1535
- bad outlier errors raw: 288
- bad core errors raw: 21
- good->medium raw: 341
- medium->good raw: 881
- nonbad->bad raw: 4

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featuregate_top20_shift_stress_goodguard_a050_bneg/original_error_waveform_panels.png)
