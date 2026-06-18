# Original Candidate Error Audit: featuregate_top20_shift_stress_a050_b0

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.831426 | 0.859341 | 0.861726 | 0.257908 | 491 | 529 | 121 |
| raw | original_all_10s+ | 0.851226 | 0.797688 | 0.899605 | 0.926585 | 3411 | 962 | 202 |
| raw | bad_core_nearboundary | 0.789916 | 0.000000 | 0.000000 | 0.789916 | 0 | 0 | 25 |
| raw | bad_outlier_stress | 0.041096 | 0.000000 | 0.000000 | 0.041096 | 0 | 0 | 96 |
| badcal | original_test_all_10s+ | 0.804294 | 0.849451 | 0.805920 | 0.386861 | 466 | 436 | 78 |
| badcal | original_all_10s+ | 0.841698 | 0.793053 | 0.870437 | 0.940776 | 3340 | 867 | 138 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.136986 | 0.000000 | 0.000000 | 0.136986 | 0 | 0 | 78 |

## Error Counts

- test errors raw: 1429
- bad outlier errors raw: 280
- bad core errors raw: 25
- good->medium raw: 491
- medium->good raw: 529
- nonbad->bad raw: 104

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featuregate_top20_shift_stress_a050_b0/original_error_waveform_panels.png)
