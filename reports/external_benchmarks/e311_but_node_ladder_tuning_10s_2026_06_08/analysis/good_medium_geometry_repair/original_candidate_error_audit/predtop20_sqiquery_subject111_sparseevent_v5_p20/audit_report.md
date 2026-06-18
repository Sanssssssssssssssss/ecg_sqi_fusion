# Original Candidate Error Audit: predtop20_sqiquery_subject111_sparseevent_v5_p20

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.798278 | 0.823901 | 0.816991 | 0.369830 | 523 | 529 | 64 |
| raw | original_all_10s+ | 0.813600 | 0.724990 | 0.892736 | 0.940208 | 4514 | 790 | 120 |
| raw | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| raw | bad_outlier_stress | 0.113014 | 0.000000 | 0.000000 | 0.113014 | 0 | 0 | 64 |
| badcal | original_test_all_10s+ | 0.788958 | 0.814011 | 0.805468 | 0.389294 | 520 | 508 | 62 |
| badcal | original_all_10s+ | 0.810201 | 0.722467 | 0.885303 | 0.942100 | 4504 | 769 | 116 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.140411 | 0.000000 | 0.000000 | 0.140411 | 0 | 0 | 62 |

## Error Counts

- test errors raw: 1710
- bad outlier errors raw: 259
- bad core errors raw: 0
- good->medium raw: 523
- medium->good raw: 529
- nonbad->bad raw: 399

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/predtop20_sqiquery_subject111_sparseevent_v5_p20/original_error_waveform_panels.png)
