# Original Candidate Error Audit: predtop20_sqiquery_subject111_burstdrop_dual_p16

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.797334 | 0.869780 | 0.781066 | 0.330900 | 420 | 675 | 68 |
| raw | original_all_10s+ | 0.824948 | 0.757613 | 0.877870 | 0.935667 | 3937 | 937 | 133 |
| raw | bad_core_nearboundary | 0.924370 | 0.000000 | 0.000000 | 0.924370 | 0 | 0 | 9 |
| raw | bad_outlier_stress | 0.089041 | 0.000000 | 0.000000 | 0.089041 | 0 | 0 | 59 |
| badcal | original_test_all_10s+ | 0.777280 | 0.846978 | 0.755310 | 0.396594 | 406 | 600 | 54 |
| badcal | original_all_10s+ | 0.815390 | 0.746230 | 0.863192 | 0.942289 | 3914 | 844 | 111 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.150685 | 0.000000 | 0.000000 | 0.150685 | 0 | 0 | 54 |

## Error Counts

- test errors raw: 1718
- bad outlier errors raw: 266
- bad core errors raw: 9
- good->medium raw: 420
- medium->good raw: 675
- nonbad->bad raw: 348

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/predtop20_sqiquery_subject111_burstdrop_dual_p16/original_error_waveform_panels.png)
