# Original Candidate Error Audit: predtop20_sqiquery_subject111_impulsebad_dual_p20

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.822225 | 0.857967 | 0.844103 | 0.270073 | 512 | 667 | 110 |
| raw | original_all_10s+ | 0.827952 | 0.744763 | 0.910331 | 0.930558 | 4340 | 923 | 176 |
| raw | bad_core_nearboundary | 0.899160 | 0.000000 | 0.000000 | 0.899160 | 0 | 0 | 12 |
| raw | bad_outlier_stress | 0.013699 | 0.000000 | 0.000000 | 0.013699 | 0 | 0 | 98 |
| badcal | original_test_all_10s+ | 0.819158 | 0.857143 | 0.834162 | 0.321168 | 509 | 659 | 89 |
| badcal | original_all_10s+ | 0.825373 | 0.744470 | 0.899793 | 0.936613 | 4332 | 915 | 144 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.044521 | 0.000000 | 0.000000 | 0.044521 | 0 | 0 | 89 |

## Error Counts

- test errors raw: 1507
- bad outlier errors raw: 288
- bad core errors raw: 12
- good->medium raw: 512
- medium->good raw: 667
- nonbad->bad raw: 28

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/predtop20_sqiquery_subject111_impulsebad_dual_p20/original_error_waveform_panels.png)
