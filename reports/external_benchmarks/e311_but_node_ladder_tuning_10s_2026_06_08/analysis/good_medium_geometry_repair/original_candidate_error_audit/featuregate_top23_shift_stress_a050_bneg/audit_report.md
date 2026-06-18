# Original Candidate Error Audit: featuregate_top23_shift_stress_a050_bneg

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.823876 | 0.865385 | 0.830095 | 0.389294 | 464 | 561 | 65 |
| raw | original_all_10s+ | 0.816331 | 0.722936 | 0.904498 | 0.940208 | 4659 | 771 | 128 |
| raw | bad_core_nearboundary | 0.991597 | 0.000000 | 0.000000 | 0.991597 | 0 | 0 | 1 |
| raw | bad_outlier_stress | 0.143836 | 0.000000 | 0.000000 | 0.143836 | 0 | 0 | 64 |
| badcal | original_test_all_10s+ | 0.822225 | 0.865110 | 0.826706 | 0.394161 | 463 | 559 | 63 |
| badcal | original_all_10s+ | 0.815542 | 0.722877 | 0.901863 | 0.940776 | 4655 | 769 | 125 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.147260 | 0.000000 | 0.000000 | 0.147260 | 0 | 0 | 63 |

## Error Counts

- test errors raw: 1493
- bad outlier errors raw: 250
- bad core errors raw: 1
- good->medium raw: 464
- medium->good raw: 561
- nonbad->bad raw: 217

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featuregate_top23_shift_stress_a050_bneg/original_error_waveform_panels.png)
