# Original Candidate Error Audit: predtop20_sqiquery_subject111_mediumonly_shift_stress

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.805592 | 0.911813 | 0.765251 | 0.299270 | 321 | 1028 | 69 |
| raw | original_all_10s+ | 0.868643 | 0.893739 | 0.796387 | 0.933018 | 1811 | 2140 | 133 |
| raw | bad_core_nearboundary | 0.966387 | 0.000000 | 0.000000 | 0.966387 | 0 | 0 | 4 |
| raw | bad_outlier_stress | 0.027397 | 0.000000 | 0.000000 | 0.027397 | 0 | 0 | 65 |
| badcal | original_test_all_10s+ | 0.786953 | 0.909066 | 0.723452 | 0.389294 | 301 | 942 | 48 |
| badcal | original_all_10s+ | 0.860663 | 0.891216 | 0.771265 | 0.941911 | 1743 | 2044 | 103 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.140411 | 0.000000 | 0.000000 | 0.140411 | 0 | 0 | 48 |

## Error Counts

- test errors raw: 1648
- bad outlier errors raw: 284
- bad core errors raw: 4
- good->medium raw: 321
- medium->good raw: 1028
- nonbad->bad raw: 11

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/predtop20_sqiquery_subject111_mediumonly_shift_stress/original_error_waveform_panels.png)
