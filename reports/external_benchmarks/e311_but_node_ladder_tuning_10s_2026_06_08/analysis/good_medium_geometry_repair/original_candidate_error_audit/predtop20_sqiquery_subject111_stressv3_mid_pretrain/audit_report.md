# Original Candidate Error Audit: predtop20_sqiquery_subject111_stressv3_mid_pretrain

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.782706 | 0.813462 | 0.794623 | 0.381995 | 499 | 409 | 64 |
| raw | original_all_10s+ | 0.807319 | 0.720589 | 0.880316 | 0.940208 | 4461 | 686 | 124 |
| raw | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| raw | bad_outlier_stress | 0.130137 | 0.000000 | 0.000000 | 0.130137 | 0 | 0 | 64 |
| badcal | original_test_all_10s+ | 0.736817 | 0.776099 | 0.731586 | 0.445255 | 425 | 342 | 49 |
| badcal | original_all_10s+ | 0.791965 | 0.709852 | 0.847196 | 0.945695 | 4316 | 615 | 106 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.219178 | 0.000000 | 0.000000 | 0.219178 | 0 | 0 | 49 |

## Error Counts

- test errors raw: 1842
- bad outlier errors raw: 254
- bad core errors raw: 0
- good->medium raw: 499
- medium->good raw: 409
- nonbad->bad raw: 680

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/predtop20_sqiquery_subject111_stressv3_mid_pretrain/original_error_waveform_panels.png)
