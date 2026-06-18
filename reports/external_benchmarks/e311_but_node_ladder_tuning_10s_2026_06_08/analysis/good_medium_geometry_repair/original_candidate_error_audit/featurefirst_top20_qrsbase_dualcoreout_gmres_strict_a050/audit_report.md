# Original Candidate Error Audit: featurefirst_top20_qrsbase_dualcoreout_gmres_strict_a050

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.846290 | 0.901648 | 0.837325 | 0.452555 | 355 | 603 | 62 |
| raw | original_all_10s+ | 0.869857 | 0.841049 | 0.878717 | 0.944939 | 2705 | 1164 | 127 |
| raw | bad_core_nearboundary | 0.949580 | 0.000000 | 0.000000 | 0.949580 | 0 | 0 | 6 |
| raw | bad_outlier_stress | 0.250000 | 0.000000 | 0.000000 | 0.250000 | 0 | 0 | 56 |
| badcal | original_test_all_10s+ | 0.846644 | 0.901648 | 0.837099 | 0.462287 | 355 | 603 | 62 |
| badcal | original_all_10s+ | 0.869948 | 0.841049 | 0.878623 | 0.945695 | 2705 | 1164 | 127 |
| badcal | bad_core_nearboundary | 0.949580 | 0.000000 | 0.000000 | 0.949580 | 0 | 0 | 6 |
| badcal | bad_outlier_stress | 0.263699 | 0.000000 | 0.000000 | 0.263699 | 0 | 0 | 56 |

## Error Counts

- test errors raw: 1303
- bad outlier errors raw: 219
- bad core errors raw: 6
- good->medium raw: 355
- medium->good raw: 603
- nonbad->bad raw: 120

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featurefirst_top20_qrsbase_dualcoreout_gmres_strict_a050/original_error_waveform_panels.png)
