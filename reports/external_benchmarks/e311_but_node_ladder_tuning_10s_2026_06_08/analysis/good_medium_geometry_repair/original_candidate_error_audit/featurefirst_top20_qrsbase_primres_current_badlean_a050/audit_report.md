# Original Candidate Error Audit: featurefirst_top20_qrsbase_primres_current_badlean_a050

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.830600 | 0.904670 | 0.802305 | 0.479319 | 341 | 748 | 49 |
| raw | original_all_10s+ | 0.866064 | 0.841460 | 0.865073 | 0.947398 | 2693 | 1297 | 112 |
| raw | bad_core_nearboundary | 0.941176 | 0.000000 | 0.000000 | 0.941176 | 0 | 0 | 7 |
| raw | bad_outlier_stress | 0.291096 | 0.000000 | 0.000000 | 0.291096 | 0 | 0 | 42 |
| badcal | original_test_all_10s+ | 0.830600 | 0.904670 | 0.802079 | 0.481752 | 341 | 747 | 48 |
| badcal | original_all_10s+ | 0.866064 | 0.841460 | 0.864979 | 0.947588 | 2693 | 1296 | 111 |
| badcal | bad_core_nearboundary | 0.949580 | 0.000000 | 0.000000 | 0.949580 | 0 | 0 | 6 |
| badcal | bad_outlier_stress | 0.291096 | 0.000000 | 0.000000 | 0.291096 | 0 | 0 | 42 |

## Error Counts

- test errors raw: 1436
- bad outlier errors raw: 207
- bad core errors raw: 7
- good->medium raw: 341
- medium->good raw: 748
- nonbad->bad raw: 133

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featurefirst_top20_qrsbase_primres_current_badlean_a050/original_error_waveform_panels.png)
