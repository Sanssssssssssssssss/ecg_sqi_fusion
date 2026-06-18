# Original Candidate Error Audit: featurefirst_top20_p20_a035

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.785537 | 0.881593 | 0.753502 | 0.279805 | 431 | 1084 | 61 |
| raw | original_all_10s+ | 0.832201 | 0.781905 | 0.863286 | 0.931883 | 3717 | 1440 | 124 |
| raw | bad_core_nearboundary | 0.957983 | 0.000000 | 0.000000 | 0.957983 | 0 | 0 | 5 |
| raw | bad_outlier_stress | 0.003425 | 0.000000 | 0.000000 | 0.003425 | 0 | 0 | 56 |
| badcal | original_test_all_10s+ | 0.783414 | 0.880220 | 0.748305 | 0.304136 | 430 | 1076 | 54 |
| badcal | original_all_10s+ | 0.831442 | 0.781553 | 0.859898 | 0.935099 | 3716 | 1432 | 110 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.020548 | 0.000000 | 0.000000 | 0.020548 | 0 | 0 | 54 |

## Error Counts

- test errors raw: 1818
- bad outlier errors raw: 291
- bad core errors raw: 5
- good->medium raw: 431
- medium->good raw: 1084
- nonbad->bad raw: 7

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featurefirst_top20_p20_a035/original_error_waveform_panels.png)
