# Original Candidate Error Audit: predtop20_sqiquery_subject111_burstdrop_p26_mguard

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.793205 | 0.900549 | 0.748983 | 0.318735 | 346 | 978 | 58 |
| raw | original_all_10s+ | 0.843792 | 0.814176 | 0.845973 | 0.934910 | 3131 | 1458 | 121 |
| raw | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| raw | bad_outlier_stress | 0.041096 | 0.000000 | 0.000000 | 0.041096 | 0 | 0 | 58 |
| badcal | original_test_all_10s+ | 0.779521 | 0.897527 | 0.724582 | 0.326034 | 334 | 961 | 57 |
| badcal | original_all_10s+ | 0.838724 | 0.813002 | 0.831483 | 0.936235 | 3107 | 1441 | 116 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.051370 | 0.000000 | 0.000000 | 0.051370 | 0 | 0 | 57 |

## Error Counts

- test errors raw: 1753
- bad outlier errors raw: 280
- bad core errors raw: 0
- good->medium raw: 346
- medium->good raw: 978
- nonbad->bad raw: 149

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/predtop20_sqiquery_subject111_burstdrop_p26_mguard/original_error_waveform_panels.png)
