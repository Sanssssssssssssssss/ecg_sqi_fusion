# Original Candidate Error Audit: predtop20_sqiquery_subject111_shift_pretrain

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.782470 | 0.884890 | 0.753276 | 0.189781 | 419 | 1090 | 112 |
| raw | original_all_10s+ | 0.812447 | 0.741829 | 0.869872 | 0.924693 | 4400 | 1377 | 175 |
| raw | bad_core_nearboundary | 0.655462 | 0.000000 | 0.000000 | 0.655462 | 0 | 0 | 41 |
| raw | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 71 |
| badcal | original_test_all_10s+ | 0.786127 | 0.884615 | 0.751243 | 0.289538 | 419 | 1086 | 72 |
| badcal | original_all_10s+ | 0.813205 | 0.741712 | 0.868461 | 0.932640 | 4400 | 1373 | 134 |
| badcal | bad_core_nearboundary | 0.949580 | 0.000000 | 0.000000 | 0.949580 | 0 | 0 | 6 |
| badcal | bad_outlier_stress | 0.020548 | 0.000000 | 0.000000 | 0.020548 | 0 | 0 | 66 |

## Error Counts

- test errors raw: 1844
- bad outlier errors raw: 292
- bad core errors raw: 41
- good->medium raw: 419
- medium->good raw: 1090
- nonbad->bad raw: 2

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/predtop20_sqiquery_subject111_shift_pretrain/original_error_waveform_panels.png)
