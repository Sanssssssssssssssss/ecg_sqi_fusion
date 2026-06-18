# Original Candidate Error Audit: featurefirst_top20_hardrec_record111stress_a050

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.820219 | 0.888187 | 0.815635 | 0.267640 | 397 | 754 | 84 |
| raw | original_all_10s+ | 0.836722 | 0.772517 | 0.893113 | 0.930369 | 3854 | 1061 | 149 |
| raw | bad_core_nearboundary | 0.882353 | 0.000000 | 0.000000 | 0.882353 | 0 | 0 | 14 |
| raw | bad_outlier_stress | 0.017123 | 0.000000 | 0.000000 | 0.017123 | 0 | 0 | 70 |
| badcal | original_test_all_10s+ | 0.800637 | 0.871703 | 0.781292 | 0.379562 | 389 | 477 | 62 |
| badcal | original_all_10s+ | 0.824493 | 0.756968 | 0.875141 | 0.940397 | 3825 | 743 | 121 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.126712 | 0.000000 | 0.000000 | 0.126712 | 0 | 0 | 62 |

## Error Counts

- test errors raw: 1524
- bad outlier errors raw: 287
- bad core errors raw: 14
- good->medium raw: 397
- medium->good raw: 754
- nonbad->bad raw: 72

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featurefirst_top20_hardrec_record111stress_a050/original_error_waveform_panels.png)
