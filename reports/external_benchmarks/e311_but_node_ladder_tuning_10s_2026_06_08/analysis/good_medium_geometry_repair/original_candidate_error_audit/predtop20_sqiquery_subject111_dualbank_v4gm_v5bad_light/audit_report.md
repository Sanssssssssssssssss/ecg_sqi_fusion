# Original Candidate Error Audit: predtop20_sqiquery_subject111_dualbank_v4gm_v5bad_light

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.776690 | 0.896429 | 0.722323 | 0.301703 | 370 | 1179 | 70 |
| raw | original_all_10s+ | 0.810080 | 0.743238 | 0.855758 | 0.933775 | 4355 | 1463 | 132 |
| raw | bad_core_nearboundary | 0.991597 | 0.000000 | 0.000000 | 0.991597 | 0 | 0 | 1 |
| raw | bad_outlier_stress | 0.020548 | 0.000000 | 0.000000 | 0.020548 | 0 | 0 | 69 |
| badcal | original_test_all_10s+ | 0.773269 | 0.890934 | 0.717578 | 0.330900 | 370 | 1135 | 64 |
| badcal | original_all_10s+ | 0.807289 | 0.739482 | 0.851336 | 0.937370 | 4355 | 1408 | 119 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.058219 | 0.000000 | 0.000000 | 0.058219 | 0 | 0 | 64 |

## Error Counts

- test errors raw: 1893
- bad outlier errors raw: 286
- bad core errors raw: 1
- good->medium raw: 370
- medium->good raw: 1179
- nonbad->bad raw: 57

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/predtop20_sqiquery_subject111_dualbank_v4gm_v5bad_light/original_error_waveform_panels.png)
