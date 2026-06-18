# Original Candidate Error Audit: predtop20_sqiquery_qrsstressv3_pretrain

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.790020 | 0.882418 | 0.775644 | 0.126521 | 428 | 989 | 135 |
| raw | original_all_10s+ | 0.803222 | 0.717831 | 0.882668 | 0.918827 | 4808 | 1239 | 201 |
| raw | bad_core_nearboundary | 0.436975 | 0.000000 | 0.000000 | 0.436975 | 0 | 0 | 67 |
| raw | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 68 |
| badcal | original_test_all_10s+ | 0.794149 | 0.878571 | 0.767510 | 0.333333 | 428 | 873 | 66 |
| badcal | original_all_10s+ | 0.801402 | 0.712551 | 0.876647 | 0.936613 | 4807 | 1092 | 126 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.061644 | 0.000000 | 0.000000 | 0.061644 | 0 | 0 | 66 |

## Error Counts

- test errors raw: 1780
- bad outlier errors raw: 292
- bad core errors raw: 67
- good->medium raw: 428
- medium->good raw: 989
- nonbad->bad raw: 4

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/predtop20_sqiquery_qrsstressv3_pretrain/original_error_waveform_panels.png)
