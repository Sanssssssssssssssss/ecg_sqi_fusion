# Original Candidate Error Audit: predtop20_sqiquery_subject111_dualbank_v4gm_v5bad

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.448626 | 0.398077 | 0.451875 | 0.861314 | 243 | 152 | 26 |
| raw | original_all_10s+ | 0.685581 | 0.652761 | 0.591268 | 0.981079 | 2728 | 873 | 69 |
| raw | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| raw | bad_outlier_stress | 0.804795 | 0.000000 | 0.000000 | 0.804795 | 0 | 0 | 26 |
| badcal | original_test_all_10s+ | 0.448626 | 0.398077 | 0.451875 | 0.861314 | 243 | 152 | 26 |
| badcal | original_all_10s+ | 0.685581 | 0.652761 | 0.591268 | 0.981079 | 2728 | 873 | 69 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.804795 | 0.000000 | 0.000000 | 0.804795 | 0 | 0 | 26 |

## Error Counts

- test errors raw: 4674
- bad outlier errors raw: 57
- bad core errors raw: 0
- good->medium raw: 243
- medium->good raw: 152
- nonbad->bad raw: 4222

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/predtop20_sqiquery_subject111_dualbank_v4gm_v5bad/original_error_waveform_panels.png)
