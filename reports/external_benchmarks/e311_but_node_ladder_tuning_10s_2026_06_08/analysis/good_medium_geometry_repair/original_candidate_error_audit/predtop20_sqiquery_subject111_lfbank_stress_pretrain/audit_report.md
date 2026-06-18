# Original Candidate Error Audit: predtop20_sqiquery_subject111_lfbank_stress_pretrain

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.786953 | 0.900275 | 0.741301 | 0.274939 | 362 | 1134 | 68 |
| raw | original_all_10s+ | 0.844581 | 0.815525 | 0.848043 | 0.931315 | 3143 | 1595 | 131 |
| raw | bad_core_nearboundary | 0.932773 | 0.000000 | 0.000000 | 0.932773 | 0 | 0 | 8 |
| raw | bad_outlier_stress | 0.006849 | 0.000000 | 0.000000 | 0.006849 | 0 | 0 | 60 |
| badcal | original_test_all_10s+ | 0.779285 | 0.896429 | 0.723226 | 0.345499 | 361 | 1084 | 51 |
| badcal | original_all_10s+ | 0.841091 | 0.814176 | 0.835999 | 0.938127 | 3139 | 1543 | 107 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.078767 | 0.000000 | 0.000000 | 0.078767 | 0 | 0 | 51 |

## Error Counts

- test errors raw: 1806
- bad outlier errors raw: 290
- bad core errors raw: 8
- good->medium raw: 362
- medium->good raw: 1134
- nonbad->bad raw: 12

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/predtop20_sqiquery_subject111_lfbank_stress_pretrain/original_error_waveform_panels.png)
