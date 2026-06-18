# Original Candidate Error Audit: predtop20_sqiquery_primtree_teacher_pretrain

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.777516 | 0.897253 | 0.730230 | 0.226277 | 374 | 1192 | 90 |
| raw | original_all_10s+ | 0.810353 | 0.743062 | 0.860275 | 0.926963 | 4379 | 1482 | 154 |
| raw | bad_core_nearboundary | 0.781513 | 0.000000 | 0.000000 | 0.781513 | 0 | 0 | 26 |
| raw | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 64 |
| badcal | original_test_all_10s+ | 0.769612 | 0.893681 | 0.709896 | 0.313869 | 371 | 1103 | 61 |
| badcal | original_all_10s+ | 0.807289 | 0.741184 | 0.849454 | 0.935667 | 4376 | 1378 | 118 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.034247 | 0.000000 | 0.000000 | 0.034247 | 0 | 0 | 61 |

## Error Counts

- test errors raw: 1886
- bad outlier errors raw: 292
- bad core errors raw: 26
- good->medium raw: 374
- medium->good raw: 1192
- nonbad->bad raw: 2

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/predtop20_sqiquery_primtree_teacher_pretrain/original_error_waveform_panels.png)
