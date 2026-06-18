# Original Candidate Error Audit: featurefirst_top20_rawbeat_artifact_auxctx_eventlocal_badrecall_a050

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.842987 | 0.870879 | 0.866697 | 0.340633 | 461 | 535 | 96 |
| raw | original_all_10s+ | 0.820367 | 0.719709 | 0.923598 | 0.937370 | 4757 | 745 | 154 |
| raw | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| raw | bad_outlier_stress | 0.071918 | 0.000000 | 0.000000 | 0.071918 | 0 | 0 | 96 |
| badcal | original_test_all_10s+ | 0.832488 | 0.870604 | 0.838003 | 0.435523 | 455 | 514 | 68 |
| badcal | original_all_10s+ | 0.816422 | 0.719592 | 0.907320 | 0.945885 | 4685 | 723 | 121 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.205479 | 0.000000 | 0.000000 | 0.205479 | 0 | 0 | 68 |

## Error Counts

- test errors raw: 1331
- bad outlier errors raw: 271
- bad core errors raw: 0
- good->medium raw: 461
- medium->good raw: 535
- nonbad->bad raw: 64

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featurefirst_top20_rawbeat_artifact_auxctx_eventlocal_badrecall_a050/original_error_waveform_panels.png)
