# Original Candidate Error Audit: predtop20_sqiquery_thresholdtree_balanced_pretrain

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.788486 | 0.898352 | 0.746724 | 0.265207 | 370 | 1112 | 66 |
| raw | original_all_10s+ | 0.821064 | 0.760430 | 0.864791 | 0.928666 | 4083 | 1423 | 139 |
| raw | bad_core_nearboundary | 0.899160 | 0.000000 | 0.000000 | 0.899160 | 0 | 0 | 12 |
| raw | bad_outlier_stress | 0.006849 | 0.000000 | 0.000000 | 0.006849 | 0 | 0 | 54 |
| badcal | original_test_all_10s+ | 0.787071 | 0.894780 | 0.740850 | 0.330900 | 370 | 1029 | 53 |
| badcal | original_all_10s+ | 0.819851 | 0.758259 | 0.860369 | 0.936991 | 4082 | 1325 | 109 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.058219 | 0.000000 | 0.000000 | 0.058219 | 0 | 0 | 53 |

## Error Counts

- test errors raw: 1793
- bad outlier errors raw: 290
- bad core errors raw: 12
- good->medium raw: 370
- medium->good raw: 1112
- nonbad->bad raw: 9

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/predtop20_sqiquery_thresholdtree_balanced_pretrain/original_error_waveform_panels.png)
