# Original Candidate Error Audit: featurefirst_top20_qrsstressv3_stress_a035

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.798042 | 0.870330 | 0.794397 | 0.197080 | 468 | 890 | 135 |
| raw | original_all_10s+ | 0.779403 | 0.660799 | 0.897817 | 0.923746 | 5776 | 1060 | 203 |
| raw | bad_core_nearboundary | 0.655462 | 0.000000 | 0.000000 | 0.655462 | 0 | 0 | 41 |
| raw | bad_outlier_stress | 0.010274 | 0.000000 | 0.000000 | 0.010274 | 0 | 0 | 94 |
| badcal | original_test_all_10s+ | 0.794975 | 0.869505 | 0.780163 | 0.294404 | 463 | 843 | 97 |
| badcal | original_all_10s+ | 0.777097 | 0.660388 | 0.886150 | 0.934153 | 5766 | 1008 | 154 |
| badcal | bad_core_nearboundary | 0.932773 | 0.000000 | 0.000000 | 0.932773 | 0 | 0 | 8 |
| badcal | bad_outlier_stress | 0.034247 | 0.000000 | 0.000000 | 0.034247 | 0 | 0 | 89 |

## Error Counts

- test errors raw: 1712
- bad outlier errors raw: 289
- bad core errors raw: 41
- good->medium raw: 468
- medium->good raw: 890
- nonbad->bad raw: 24

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featurefirst_top20_qrsstressv3_stress_a035/original_error_waveform_panels.png)
