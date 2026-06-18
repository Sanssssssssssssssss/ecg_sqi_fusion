# Original Candidate Error Audit: featurefirst_top20_rawbeat_qfeatbin_a050

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.792851 | 0.893681 | 0.756439 | 0.291971 | 387 | 1071 | 120 |
| raw | original_all_10s+ | 0.820973 | 0.754679 | 0.871942 | 0.932261 | 4181 | 1350 | 185 |
| raw | bad_core_nearboundary | 0.924370 | 0.000000 | 0.000000 | 0.924370 | 0 | 0 | 9 |
| raw | bad_outlier_stress | 0.034247 | 0.000000 | 0.000000 | 0.034247 | 0 | 0 | 111 |
| badcal | original_test_all_10s+ | 0.793323 | 0.892033 | 0.747854 | 0.408759 | 387 | 1025 | 83 |
| badcal | original_all_10s+ | 0.820973 | 0.754210 | 0.867332 | 0.943046 | 4179 | 1298 | 140 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.167808 | 0.000000 | 0.000000 | 0.167808 | 0 | 0 | 83 |

## Error Counts

- test errors raw: 1756
- bad outlier errors raw: 282
- bad core errors raw: 9
- good->medium raw: 387
- medium->good raw: 1071
- nonbad->bad raw: 7

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featurefirst_top20_rawbeat_qfeatbin_a050/original_error_waveform_panels.png)
