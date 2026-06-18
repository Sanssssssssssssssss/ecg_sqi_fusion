# Original Candidate Error Audit: predtop20_sqiquery_primtree_stress_teacher_pretrain

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.777398 | 0.903571 | 0.723452 | 0.240876 | 351 | 1224 | 67 |
| raw | original_all_10s+ | 0.819487 | 0.766297 | 0.850772 | 0.928098 | 3983 | 1586 | 132 |
| raw | bad_core_nearboundary | 0.831933 | 0.000000 | 0.000000 | 0.831933 | 0 | 0 | 20 |
| raw | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 47 |
| badcal | original_test_all_10s+ | 0.779049 | 0.903571 | 0.721645 | 0.294404 | 351 | 1218 | 46 |
| badcal | original_all_10s+ | 0.819396 | 0.766297 | 0.847667 | 0.933775 | 3983 | 1577 | 104 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.006849 | 0.000000 | 0.000000 | 0.006849 | 0 | 0 | 46 |

## Error Counts

- test errors raw: 1887
- bad outlier errors raw: 292
- bad core errors raw: 20
- good->medium raw: 351
- medium->good raw: 1224
- nonbad->bad raw: 0

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/predtop20_sqiquery_primtree_stress_teacher_pretrain/original_error_waveform_panels.png)
