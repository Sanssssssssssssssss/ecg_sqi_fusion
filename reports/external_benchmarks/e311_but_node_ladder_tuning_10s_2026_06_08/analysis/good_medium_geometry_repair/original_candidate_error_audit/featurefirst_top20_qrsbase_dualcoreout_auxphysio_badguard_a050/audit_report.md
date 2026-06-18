# Original Candidate Error Audit: featurefirst_top20_qrsbase_dualcoreout_auxphysio_badguard_a050

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.795564 | 0.920330 | 0.739042 | 0.299270 | 286 | 1118 | 56 |
| raw | original_all_10s+ | 0.864334 | 0.873144 | 0.815958 | 0.933207 | 2156 | 1911 | 119 |
| raw | bad_core_nearboundary | 0.949580 | 0.000000 | 0.000000 | 0.949580 | 0 | 0 | 6 |
| raw | bad_outlier_stress | 0.034247 | 0.000000 | 0.000000 | 0.034247 | 0 | 0 | 50 |
| badcal | original_test_all_10s+ | 0.798042 | 0.919780 | 0.723904 | 0.518248 | 282 | 1029 | 30 |
| badcal | original_all_10s+ | 0.864638 | 0.872968 | 0.808148 | 0.951372 | 2147 | 1821 | 88 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.321918 | 0.000000 | 0.000000 | 0.321918 | 0 | 0 | 30 |

## Error Counts

- test errors raw: 1733
- bad outlier errors raw: 282
- bad core errors raw: 6
- good->medium raw: 286
- medium->good raw: 1118
- nonbad->bad raw: 41

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featurefirst_top20_qrsbase_dualcoreout_auxphysio_badguard_a050/original_error_waveform_panels.png)
