# Original Candidate Error Audit: predtop20_sqiquery_subject111_recordbad_strong_pretrain

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.750383 | 0.879670 | 0.708766 | 0.053528 | 369 | 795 | 166 |
| raw | original_all_10s+ | 0.663521 | 0.759725 | 0.837128 | 0.004163 | 3929 | 1199 | 5038 |
| raw | bad_core_nearboundary | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 119 |
| raw | bad_outlier_stress | 0.075342 | 0.000000 | 0.000000 | 0.075342 | 0 | 0 | 47 |
| badcal | original_test_all_10s+ | 0.716527 | 0.810989 | 0.692725 | 0.136253 | 364 | 523 | 165 |
| badcal | original_all_10s+ | 0.648592 | 0.733556 | 0.829507 | 0.010785 | 3922 | 884 | 5037 |
| badcal | bad_core_nearboundary | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 119 |
| badcal | bad_outlier_stress | 0.191781 | 0.000000 | 0.000000 | 0.191781 | 0 | 0 | 46 |

## Error Counts

- test errors raw: 2116
- bad outlier errors raw: 270
- bad core errors raw: 119
- good->medium raw: 369
- medium->good raw: 795
- nonbad->bad raw: 563

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/predtop20_sqiquery_subject111_recordbad_strong_pretrain/original_error_waveform_panels.png)
