# Original Candidate Error Audit: predtop20_qrsbank_patch_pretrain

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.784830 | 0.912363 | 0.739494 | 0.143552 | 319 | 1153 | 125 |
| raw | original_all_10s+ | 0.826921 | 0.774864 | 0.863474 | 0.921287 | 3837 | 1451 | 187 |
| raw | bad_core_nearboundary | 0.495798 | 0.000000 | 0.000000 | 0.495798 | 0 | 0 | 60 |
| raw | bad_outlier_stress | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 65 |
| badcal | original_test_all_10s+ | 0.789666 | 0.906593 | 0.732716 | 0.367397 | 318 | 919 | 63 |
| badcal | original_all_10s+ | 0.823006 | 0.765358 | 0.857452 | 0.939640 | 3833 | 1178 | 121 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.109589 | 0.000000 | 0.000000 | 0.109589 | 0 | 0 | 63 |

## Error Counts

- test errors raw: 1824
- bad outlier errors raw: 292
- bad core errors raw: 60
- good->medium raw: 319
- medium->good raw: 1153
- nonbad->bad raw: 0

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/predtop20_qrsbank_patch_pretrain/original_error_waveform_panels.png)
