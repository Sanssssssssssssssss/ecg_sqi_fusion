# Original Candidate Error Audit: featurefirst_top20_rawbeat_artifact_auxctx_physioctx_badrecall_a050

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.840981 | 0.864560 | 0.868504 | 0.335766 | 484 | 531 | 96 |
| raw | original_all_10s+ | 0.814025 | 0.706390 | 0.925480 | 0.936991 | 4984 | 729 | 154 |
| raw | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| raw | bad_outlier_stress | 0.065068 | 0.000000 | 0.000000 | 0.065068 | 0 | 0 | 96 |
| badcal | original_test_all_10s+ | 0.819630 | 0.864011 | 0.819702 | 0.425791 | 472 | 485 | 69 |
| badcal | original_all_10s+ | 0.806105 | 0.705099 | 0.898946 | 0.945128 | 4867 | 679 | 121 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.191781 | 0.000000 | 0.000000 | 0.191781 | 0 | 0 | 69 |

## Error Counts

- test errors raw: 1348
- bad outlier errors raw: 273
- bad core errors raw: 0
- good->medium raw: 484
- medium->good raw: 531
- nonbad->bad raw: 60

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featurefirst_top20_rawbeat_artifact_auxctx_physioctx_badrecall_a050/original_error_waveform_panels.png)
