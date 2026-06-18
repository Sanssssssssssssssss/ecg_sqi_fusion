# Original Candidate Error Audit: p20_sqiquery_primctx_v5_badguard

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.794857 | 0.895879 | 0.757343 | 0.304136 | 376 | 1048 | 65 |
| raw | original_all_10s+ | 0.834446 | 0.785132 | 0.864038 | 0.933964 | 3657 | 1404 | 127 |
| raw | bad_core_nearboundary | 0.991597 | 0.000000 | 0.000000 | 0.991597 | 0 | 0 | 1 |
| raw | bad_outlier_stress | 0.023973 | 0.000000 | 0.000000 | 0.023973 | 0 | 0 | 64 |
| badcal | original_test_all_10s+ | 0.792025 | 0.894505 | 0.751695 | 0.318735 | 376 | 1040 | 62 |
| badcal | original_all_10s+ | 0.833050 | 0.784604 | 0.859522 | 0.936045 | 3657 | 1396 | 119 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.041096 | 0.000000 | 0.000000 | 0.041096 | 0 | 0 | 62 |

## Error Counts

- test errors raw: 1739
- bad outlier errors raw: 285
- bad core errors raw: 1
- good->medium raw: 376
- medium->good raw: 1048
- nonbad->bad raw: 29

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/p20_sqiquery_primctx_v5_badguard/original_error_waveform_panels.png)
