# Original Candidate Error Audit: featurefirst_top20_rawbeat_artifact_auxctx_thresholdpiece_badrecall_a050

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.831072 | 0.880495 | 0.836195 | 0.338200 | 426 | 671 | 85 |
| raw | original_all_10s+ | 0.826375 | 0.742710 | 0.905438 | 0.937181 | 4365 | 936 | 143 |
| raw | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| raw | bad_outlier_stress | 0.068493 | 0.000000 | 0.000000 | 0.068493 | 0 | 0 | 85 |
| badcal | original_test_all_10s+ | 0.826590 | 0.879396 | 0.820606 | 0.423358 | 421 | 640 | 66 |
| badcal | original_all_10s+ | 0.822521 | 0.742064 | 0.890760 | 0.944749 | 4311 | 904 | 119 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.188356 | 0.000000 | 0.000000 | 0.188356 | 0 | 0 | 66 |

## Error Counts

- test errors raw: 1432
- bad outlier errors raw: 272
- bad core errors raw: 0
- good->medium raw: 426
- medium->good raw: 671
- nonbad->bad raw: 63

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featurefirst_top20_rawbeat_artifact_auxctx_thresholdpiece_badrecall_a050/original_error_waveform_panels.png)
