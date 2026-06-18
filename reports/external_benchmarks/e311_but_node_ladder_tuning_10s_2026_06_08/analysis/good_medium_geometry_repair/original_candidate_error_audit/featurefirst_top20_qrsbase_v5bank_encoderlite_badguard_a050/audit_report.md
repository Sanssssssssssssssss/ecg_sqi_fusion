# Original Candidate Error Audit: featurefirst_top20_qrsbase_v5bank_encoderlite_badguard_a050

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.773977 | 0.923901 | 0.711252 | 0.121655 | 275 | 1252 | 116 |
| raw | original_all_10s+ | 0.853775 | 0.856187 | 0.818310 | 0.917313 | 2445 | 1901 | 190 |
| raw | bad_core_nearboundary | 0.361345 | 0.000000 | 0.000000 | 0.361345 | 0 | 0 | 76 |
| raw | bad_outlier_stress | 0.023973 | 0.000000 | 0.000000 | 0.023973 | 0 | 0 | 40 |
| badcal | original_test_all_10s+ | 0.784358 | 0.923901 | 0.700181 | 0.454988 | 271 | 1200 | 34 |
| badcal | original_all_10s+ | 0.856293 | 0.856187 | 0.811441 | 0.946831 | 2436 | 1849 | 90 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.232877 | 0.000000 | 0.000000 | 0.232877 | 0 | 0 | 34 |

## Error Counts

- test errors raw: 1916
- bad outlier errors raw: 285
- bad core errors raw: 76
- good->medium raw: 275
- medium->good raw: 1252
- nonbad->bad raw: 28

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featurefirst_top20_qrsbase_v5bank_encoderlite_badguard_a050/original_error_waveform_panels.png)
