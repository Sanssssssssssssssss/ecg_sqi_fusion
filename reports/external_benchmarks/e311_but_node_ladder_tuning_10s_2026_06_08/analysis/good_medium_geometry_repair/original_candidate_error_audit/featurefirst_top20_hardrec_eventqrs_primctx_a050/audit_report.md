# Original Candidate Error Audit: featurefirst_top20_hardrec_eventqrs_primctx_a050

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.829775 | 0.880495 | 0.838681 | 0.284672 | 428 | 652 | 118 |
| raw | original_all_10s+ | 0.828226 | 0.746582 | 0.907320 | 0.932450 | 4310 | 912 | 179 |
| raw | bad_core_nearboundary | 0.907563 | 0.000000 | 0.000000 | 0.907563 | 0 | 0 | 11 |
| raw | bad_outlier_stress | 0.030822 | 0.000000 | 0.000000 | 0.030822 | 0 | 0 | 107 |
| badcal | original_test_all_10s+ | 0.816091 | 0.876374 | 0.806371 | 0.386861 | 419 | 577 | 87 |
| badcal | original_all_10s+ | 0.823067 | 0.743238 | 0.892360 | 0.941154 | 4286 | 831 | 145 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.136986 | 0.000000 | 0.000000 | 0.136986 | 0 | 0 | 87 |

## Error Counts

- test errors raw: 1443
- bad outlier errors raw: 283
- bad core errors raw: 11
- good->medium raw: 428
- medium->good raw: 652
- nonbad->bad raw: 69

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featurefirst_top20_hardrec_eventqrs_primctx_a050/original_error_waveform_panels.png)
