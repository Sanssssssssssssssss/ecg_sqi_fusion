# Original Candidate Error Audit: featurefirst_wavecomp_dualcoreout_encoderlite_baselinegm_a050

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.774684 | 0.927747 | 0.683235 | 0.403893 | 259 | 1308 | 30 |
| raw | original_all_10s+ | 0.855353 | 0.871267 | 0.787072 | 0.941343 | 2188 | 2159 | 93 |
| raw | bad_core_nearboundary | 0.983193 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 2 |
| raw | bad_outlier_stress | 0.167808 | 0.000000 | 0.000000 | 0.167808 | 0 | 0 | 28 |
| badcal | original_test_all_10s+ | 0.776100 | 0.927747 | 0.683009 | 0.435523 | 259 | 1301 | 29 |
| badcal | original_all_10s+ | 0.855747 | 0.871267 | 0.786978 | 0.943992 | 2188 | 2152 | 91 |
| badcal | bad_core_nearboundary | 0.991597 | 0.000000 | 0.000000 | 0.991597 | 0 | 0 | 1 |
| badcal | bad_outlier_stress | 0.208904 | 0.000000 | 0.000000 | 0.208904 | 0 | 0 | 28 |

## Error Counts

- test errors raw: 1910
- bad outlier errors raw: 243
- bad core errors raw: 2
- good->medium raw: 259
- medium->good raw: 1308
- nonbad->bad raw: 98

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featurefirst_wavecomp_dualcoreout_encoderlite_baselinegm_a050/original_error_waveform_panels.png)
