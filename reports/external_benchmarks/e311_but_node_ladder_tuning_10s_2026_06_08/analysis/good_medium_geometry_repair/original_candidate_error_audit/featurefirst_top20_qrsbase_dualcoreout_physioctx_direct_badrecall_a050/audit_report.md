# Original Candidate Error Audit: featurefirst_top20_qrsbase_dualcoreout_physioctx_direct_badrecall_a050

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.844049 | 0.903571 | 0.826254 | 0.508516 | 347 | 627 | 72 |
| raw | original_all_10s+ | 0.865214 | 0.828258 | 0.882763 | 0.949101 | 2922 | 1096 | 138 |
| raw | bad_core_nearboundary | 0.882353 | 0.000000 | 0.000000 | 0.882353 | 0 | 0 | 14 |
| raw | bad_outlier_stress | 0.356164 | 0.000000 | 0.000000 | 0.356164 | 0 | 0 | 58 |
| badcal | original_test_all_10s+ | 0.844049 | 0.903571 | 0.826254 | 0.508516 | 347 | 627 | 72 |
| badcal | original_all_10s+ | 0.865214 | 0.828258 | 0.882763 | 0.949101 | 2922 | 1096 | 138 |
| badcal | bad_core_nearboundary | 0.882353 | 0.000000 | 0.000000 | 0.882353 | 0 | 0 | 14 |
| badcal | bad_outlier_stress | 0.356164 | 0.000000 | 0.000000 | 0.356164 | 0 | 0 | 58 |

## Error Counts

- test errors raw: 1322
- bad outlier errors raw: 188
- bad core errors raw: 14
- good->medium raw: 347
- medium->good raw: 627
- nonbad->bad raw: 146

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featurefirst_top20_qrsbase_dualcoreout_physioctx_direct_badrecall_a050/original_error_waveform_panels.png)
