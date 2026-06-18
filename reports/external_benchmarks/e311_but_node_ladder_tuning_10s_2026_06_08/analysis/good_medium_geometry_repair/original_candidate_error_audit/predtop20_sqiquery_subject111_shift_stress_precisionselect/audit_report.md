# Original Candidate Error Audit: predtop20_sqiquery_subject111_shift_stress_precisionselect

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.816680 | 0.876374 | 0.822187 | 0.228710 | 450 | 778 | 119 |
| raw | original_all_10s+ | 0.838512 | 0.783137 | 0.883609 | 0.926395 | 3695 | 1224 | 189 |
| raw | bad_core_nearboundary | 0.773109 | 0.000000 | 0.000000 | 0.773109 | 0 | 0 | 27 |
| raw | bad_outlier_stress | 0.006849 | 0.000000 | 0.000000 | 0.006849 | 0 | 0 | 92 |
| badcal | original_test_all_10s+ | 0.803350 | 0.874725 | 0.786715 | 0.350365 | 426 | 729 | 74 |
| badcal | original_all_10s+ | 0.833778 | 0.781553 | 0.865638 | 0.938127 | 3621 | 1169 | 133 |
| badcal | bad_core_nearboundary | 0.991597 | 0.000000 | 0.000000 | 0.991597 | 0 | 0 | 1 |
| badcal | bad_outlier_stress | 0.089041 | 0.000000 | 0.000000 | 0.089041 | 0 | 0 | 73 |

## Error Counts

- test errors raw: 1554
- bad outlier errors raw: 290
- bad core errors raw: 27
- good->medium raw: 450
- medium->good raw: 778
- nonbad->bad raw: 9

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/predtop20_sqiquery_subject111_shift_stress_precisionselect/original_error_waveform_panels.png)
