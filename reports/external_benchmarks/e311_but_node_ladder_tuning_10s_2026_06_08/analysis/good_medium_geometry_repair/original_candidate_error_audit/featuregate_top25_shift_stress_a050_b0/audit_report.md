# Original Candidate Error Audit: featuregate_top25_shift_stress_a050_b0

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.811726 | 0.881868 | 0.816765 | 0.136253 | 427 | 793 | 138 |
| raw | original_all_10s+ | 0.849739 | 0.808660 | 0.881633 | 0.918070 | 3256 | 1231 | 214 |
| raw | bad_core_nearboundary | 0.453782 | 0.000000 | 0.000000 | 0.453782 | 0 | 0 | 65 |
| raw | bad_outlier_stress | 0.006849 | 0.000000 | 0.000000 | 0.006849 | 0 | 0 | 73 |
| badcal | original_test_all_10s+ | 0.814085 | 0.878022 | 0.805920 | 0.335766 | 425 | 696 | 71 |
| badcal | original_all_10s+ | 0.849314 | 0.805609 | 0.875894 | 0.936802 | 3247 | 1123 | 130 |
| badcal | bad_core_nearboundary | 0.957983 | 0.000000 | 0.000000 | 0.957983 | 0 | 0 | 5 |
| badcal | bad_outlier_stress | 0.082192 | 0.000000 | 0.000000 | 0.082192 | 0 | 0 | 66 |

## Error Counts

- test errors raw: 1596
- bad outlier errors raw: 290
- bad core errors raw: 65
- good->medium raw: 427
- medium->good raw: 793
- nonbad->bad raw: 21

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featuregate_top25_shift_stress_a050_b0/original_error_waveform_panels.png)
