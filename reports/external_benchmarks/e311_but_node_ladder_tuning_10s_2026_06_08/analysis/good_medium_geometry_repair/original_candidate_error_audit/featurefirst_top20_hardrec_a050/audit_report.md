# Original Candidate Error Audit: featurefirst_top20_hardrec_a050

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.844520 | 0.848352 | 0.894261 | 0.274939 | 550 | 451 | 129 |
| raw | original_all_10s+ | 0.845976 | 0.768644 | 0.927832 | 0.930747 | 3940 | 745 | 195 |
| raw | bad_core_nearboundary | 0.932773 | 0.000000 | 0.000000 | 0.932773 | 0 | 0 | 8 |
| raw | bad_outlier_stress | 0.006849 | 0.000000 | 0.000000 | 0.006849 | 0 | 0 | 121 |
| badcal | original_test_all_10s+ | 0.838976 | 0.848077 | 0.879575 | 0.321168 | 540 | 450 | 110 |
| badcal | original_all_10s+ | 0.844399 | 0.768468 | 0.920587 | 0.936045 | 3920 | 744 | 167 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.044521 | 0.000000 | 0.000000 | 0.044521 | 0 | 0 | 110 |

## Error Counts

- test errors raw: 1318
- bad outlier errors raw: 290
- bad core errors raw: 8
- good->medium raw: 550
- medium->good raw: 451
- nonbad->bad raw: 19

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featurefirst_top20_hardrec_a050/original_error_waveform_panels.png)
