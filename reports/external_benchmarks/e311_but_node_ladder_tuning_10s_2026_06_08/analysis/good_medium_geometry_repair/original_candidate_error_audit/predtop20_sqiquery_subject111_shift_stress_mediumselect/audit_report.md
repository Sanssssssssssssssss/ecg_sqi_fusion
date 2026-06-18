# Original Candidate Error Audit: predtop20_sqiquery_subject111_shift_stress_mediumselect

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.820809 | 0.854121 | 0.843425 | 0.282238 | 505 | 585 | 112 |
| raw | original_all_10s+ | 0.811264 | 0.710673 | 0.913154 | 0.930747 | 4876 | 794 | 182 |
| raw | bad_core_nearboundary | 0.848739 | 0.000000 | 0.000000 | 0.848739 | 0 | 0 | 18 |
| raw | bad_outlier_stress | 0.051370 | 0.000000 | 0.000000 | 0.051370 | 0 | 0 | 94 |
| badcal | original_test_all_10s+ | 0.786363 | 0.822527 | 0.789652 | 0.430657 | 478 | 371 | 75 |
| badcal | original_all_10s+ | 0.797427 | 0.697706 | 0.884174 | 0.944560 | 4781 | 570 | 133 |
| badcal | bad_core_nearboundary | 0.983193 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 2 |
| badcal | bad_outlier_stress | 0.205479 | 0.000000 | 0.000000 | 0.205479 | 0 | 0 | 73 |

## Error Counts

- test errors raw: 1519
- bad outlier errors raw: 277
- bad core errors raw: 18
- good->medium raw: 505
- medium->good raw: 585
- nonbad->bad raw: 134

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/predtop20_sqiquery_subject111_shift_stress_mediumselect/original_error_waveform_panels.png)
