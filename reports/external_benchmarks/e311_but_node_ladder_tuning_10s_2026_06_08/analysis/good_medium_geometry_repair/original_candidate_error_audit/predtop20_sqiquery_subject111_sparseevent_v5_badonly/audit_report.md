# Original Candidate Error Audit: predtop20_sqiquery_subject111_sparseevent_v5_badonly

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.815501 | 0.881044 | 0.806146 | 0.335766 | 395 | 728 | 79 |
| raw | original_all_10s+ | 0.843943 | 0.799566 | 0.868461 | 0.937748 | 3348 | 1166 | 134 |
| raw | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| raw | bad_outlier_stress | 0.065068 | 0.000000 | 0.000000 | 0.065068 | 0 | 0 | 79 |
| badcal | original_test_all_10s+ | 0.812552 | 0.880220 | 0.800723 | 0.340633 | 395 | 728 | 78 |
| badcal | original_all_10s+ | 0.842457 | 0.799390 | 0.863850 | 0.938316 | 3347 | 1166 | 132 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.071918 | 0.000000 | 0.000000 | 0.071918 | 0 | 0 | 78 |

## Error Counts

- test errors raw: 1564
- bad outlier errors raw: 273
- bad core errors raw: 0
- good->medium raw: 395
- medium->good raw: 728
- nonbad->bad raw: 168

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/predtop20_sqiquery_subject111_sparseevent_v5_badonly/original_error_waveform_panels.png)
