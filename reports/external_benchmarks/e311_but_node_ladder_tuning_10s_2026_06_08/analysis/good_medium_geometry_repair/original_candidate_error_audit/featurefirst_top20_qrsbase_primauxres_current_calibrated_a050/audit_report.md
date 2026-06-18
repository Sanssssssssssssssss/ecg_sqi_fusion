# Original Candidate Error Audit: featurefirst_top20_qrsbase_primauxres_current_calibrated_a050

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.802642 | 0.915110 | 0.755084 | 0.318735 | 301 | 997 | 62 |
| raw | original_all_10s+ | 0.864577 | 0.865927 | 0.827625 | 0.934532 | 2271 | 1733 | 126 |
| raw | bad_core_nearboundary | 0.949580 | 0.000000 | 0.000000 | 0.949580 | 0 | 0 | 6 |
| raw | bad_outlier_stress | 0.061644 | 0.000000 | 0.000000 | 0.061644 | 0 | 0 | 56 |
| badcal | original_test_all_10s+ | 0.802053 | 0.915110 | 0.746272 | 0.401460 | 300 | 966 | 49 |
| badcal | original_all_10s+ | 0.864213 | 0.865927 | 0.823109 | 0.941343 | 2264 | 1702 | 112 |
| badcal | bad_core_nearboundary | 0.966387 | 0.000000 | 0.000000 | 0.966387 | 0 | 0 | 4 |
| badcal | bad_outlier_stress | 0.171233 | 0.000000 | 0.000000 | 0.171233 | 0 | 0 | 45 |

## Error Counts

- test errors raw: 1673
- bad outlier errors raw: 274
- bad core errors raw: 6
- good->medium raw: 301
- medium->good raw: 997
- nonbad->bad raw: 95

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featurefirst_top20_qrsbase_primauxres_current_calibrated_a050/original_error_waveform_panels.png)
