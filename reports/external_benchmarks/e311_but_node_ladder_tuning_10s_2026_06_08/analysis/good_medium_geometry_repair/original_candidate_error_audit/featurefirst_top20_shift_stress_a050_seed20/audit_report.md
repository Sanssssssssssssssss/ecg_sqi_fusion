# Original Candidate Error Audit: featurefirst_top20_shift_stress_a050_seed20

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.795093 | 0.900275 | 0.756213 | 0.282238 | 355 | 1051 | 61 |
| raw | original_all_10s+ | 0.854048 | 0.844100 | 0.832612 | 0.929234 | 2640 | 1735 | 138 |
| raw | bad_core_nearboundary | 0.899160 | 0.000000 | 0.000000 | 0.899160 | 0 | 0 | 12 |
| raw | bad_outlier_stress | 0.030822 | 0.000000 | 0.000000 | 0.030822 | 0 | 0 | 49 |
| badcal | original_test_all_10s+ | 0.794385 | 0.900275 | 0.752824 | 0.304136 | 353 | 1051 | 52 |
| badcal | original_all_10s+ | 0.853775 | 0.844100 | 0.829883 | 0.933018 | 2633 | 1735 | 118 |
| badcal | bad_core_nearboundary | 0.957983 | 0.000000 | 0.000000 | 0.957983 | 0 | 0 | 5 |
| badcal | bad_outlier_stress | 0.037671 | 0.000000 | 0.000000 | 0.037671 | 0 | 0 | 47 |

## Error Counts

- test errors raw: 1737
- bad outlier errors raw: 283
- bad core errors raw: 12
- good->medium raw: 355
- medium->good raw: 1051
- nonbad->bad raw: 36

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featurefirst_top20_shift_stress_a050_seed20/original_error_waveform_panels.png)
