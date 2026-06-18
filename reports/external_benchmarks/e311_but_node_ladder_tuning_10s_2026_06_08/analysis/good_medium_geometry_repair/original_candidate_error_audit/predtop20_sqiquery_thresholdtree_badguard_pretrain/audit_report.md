# Original Candidate Error Audit: predtop20_sqiquery_thresholdtree_badguard_pretrain

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.766545 | 0.887363 | 0.711026 | 0.294404 | 405 | 1236 | 54 |
| raw | original_all_10s+ | 0.776490 | 0.677170 | 0.857734 | 0.933396 | 5493 | 1459 | 114 |
| raw | bad_core_nearboundary | 0.974790 | 0.000000 | 0.000000 | 0.974790 | 0 | 0 | 3 |
| raw | bad_outlier_stress | 0.017123 | 0.000000 | 0.000000 | 0.017123 | 0 | 0 | 51 |
| badcal | original_test_all_10s+ | 0.759821 | 0.881044 | 0.698373 | 0.347932 | 402 | 1077 | 51 |
| badcal | original_all_10s+ | 0.770846 | 0.670363 | 0.848796 | 0.938127 | 5488 | 1269 | 109 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.082192 | 0.000000 | 0.000000 | 0.082192 | 0 | 0 | 51 |

## Error Counts

- test errors raw: 1979
- bad outlier errors raw: 287
- bad core errors raw: 3
- good->medium raw: 405
- medium->good raw: 1236
- nonbad->bad raw: 48

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/predtop20_sqiquery_thresholdtree_badguard_pretrain/original_error_waveform_panels.png)
