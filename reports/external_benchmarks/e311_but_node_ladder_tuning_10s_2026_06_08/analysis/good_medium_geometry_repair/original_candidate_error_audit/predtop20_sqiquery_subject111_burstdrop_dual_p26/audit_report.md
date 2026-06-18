# Original Candidate Error Audit: predtop20_sqiquery_subject111_burstdrop_dual_p26

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.816916 | 0.885440 | 0.810664 | 0.277372 | 410 | 806 | 97 |
| raw | original_all_10s+ | 0.855353 | 0.825442 | 0.865638 | 0.931126 | 2958 | 1387 | 163 |
| raw | bad_core_nearboundary | 0.924370 | 0.000000 | 0.000000 | 0.924370 | 0 | 0 | 9 |
| raw | bad_outlier_stress | 0.013699 | 0.000000 | 0.000000 | 0.013699 | 0 | 0 | 88 |
| badcal | original_test_all_10s+ | 0.812552 | 0.885165 | 0.799367 | 0.311436 | 406 | 804 | 84 |
| badcal | original_all_10s+ | 0.853653 | 0.825324 | 0.858675 | 0.934910 | 2949 | 1385 | 144 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.030822 | 0.000000 | 0.000000 | 0.030822 | 0 | 0 | 84 |

## Error Counts

- test errors raw: 1552
- bad outlier errors raw: 288
- bad core errors raw: 9
- good->medium raw: 410
- medium->good raw: 806
- nonbad->bad raw: 39

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/predtop20_sqiquery_subject111_burstdrop_dual_p26/original_error_waveform_panels.png)
