# Original Candidate Error Audit: featurefirst_top20_qrsbase_primauxres_current_featureonly_a050

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.853486 | 0.897802 | 0.837099 | 0.637470 | 365 | 529 | 55 |
| raw | original_all_10s+ | 0.864092 | 0.817638 | 0.891231 | 0.959319 | 3099 | 952 | 120 |
| raw | bad_core_nearboundary | 0.907563 | 0.000000 | 0.000000 | 0.907563 | 0 | 0 | 11 |
| raw | bad_outlier_stress | 0.527397 | 0.000000 | 0.000000 | 0.527397 | 0 | 0 | 44 |
| badcal | original_test_all_10s+ | 0.853486 | 0.897802 | 0.837099 | 0.637470 | 365 | 529 | 55 |
| badcal | original_all_10s+ | 0.864092 | 0.817638 | 0.891231 | 0.959319 | 3099 | 952 | 120 |
| badcal | bad_core_nearboundary | 0.907563 | 0.000000 | 0.000000 | 0.907563 | 0 | 0 | 11 |
| badcal | bad_outlier_stress | 0.527397 | 0.000000 | 0.000000 | 0.527397 | 0 | 0 | 44 |

## Error Counts

- test errors raw: 1242
- bad outlier errors raw: 138
- bad core errors raw: 11
- good->medium raw: 365
- medium->good raw: 529
- nonbad->bad raw: 199

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featurefirst_top20_qrsbase_primauxres_current_featureonly_a050/original_error_waveform_panels.png)
