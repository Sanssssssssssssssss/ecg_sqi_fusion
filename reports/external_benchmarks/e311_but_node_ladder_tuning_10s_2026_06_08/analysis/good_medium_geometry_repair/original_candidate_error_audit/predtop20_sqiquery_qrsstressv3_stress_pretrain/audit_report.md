# Original Candidate Error Audit: predtop20_sqiquery_qrsstressv3_stress_pretrain

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.804530 | 0.895055 | 0.781066 | 0.255474 | 377 | 928 | 111 |
| raw | original_all_10s+ | 0.825737 | 0.759432 | 0.880316 | 0.929801 | 4092 | 1218 | 173 |
| raw | bad_core_nearboundary | 0.831933 | 0.000000 | 0.000000 | 0.831933 | 0 | 0 | 20 |
| raw | bad_outlier_stress | 0.020548 | 0.000000 | 0.000000 | 0.020548 | 0 | 0 | 91 |
| badcal | original_test_all_10s+ | 0.794385 | 0.888736 | 0.755535 | 0.377129 | 371 | 733 | 83 |
| badcal | original_all_10s+ | 0.819274 | 0.753506 | 0.864697 | 0.940019 | 4078 | 996 | 143 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.123288 | 0.000000 | 0.000000 | 0.123288 | 0 | 0 | 83 |

## Error Counts

- test errors raw: 1657
- bad outlier errors raw: 286
- bad core errors raw: 20
- good->medium raw: 377
- medium->good raw: 928
- nonbad->bad raw: 46

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/predtop20_sqiquery_qrsstressv3_stress_pretrain/original_error_waveform_panels.png)
