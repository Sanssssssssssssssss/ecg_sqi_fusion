# Original Candidate Error Audit: featurefirst_top20_shift_stress_a050_seed18

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.817978 | 0.905495 | 0.792363 | 0.318735 | 336 | 863 | 63 |
| raw | original_all_10s+ | 0.866094 | 0.854251 | 0.850583 | 0.935478 | 2466 | 1491 | 122 |
| raw | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| raw | bad_outlier_stress | 0.041096 | 0.000000 | 0.000000 | 0.041096 | 0 | 0 | 63 |
| badcal | original_test_all_10s+ | 0.814321 | 0.904396 | 0.784907 | 0.333333 | 335 | 850 | 62 |
| badcal | original_all_10s+ | 0.863272 | 0.853430 | 0.842209 | 0.937370 | 2456 | 1478 | 117 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.061644 | 0.000000 | 0.000000 | 0.061644 | 0 | 0 | 62 |

## Error Counts

- test errors raw: 1543
- bad outlier errors raw: 280
- bad core errors raw: 0
- good->medium raw: 336
- medium->good raw: 863
- nonbad->bad raw: 64

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featurefirst_top20_shift_stress_a050_seed18/original_error_waveform_panels.png)
