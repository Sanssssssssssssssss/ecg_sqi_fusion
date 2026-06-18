# Original Candidate Error Audit: predtop20_sqiquery_subject111_medium_gapheavy_mguard

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.777280 | 0.912637 | 0.715545 | 0.243309 | 316 | 1238 | 139 |
| raw | original_all_10s+ | 0.841455 | 0.845039 | 0.792247 | 0.928855 | 2637 | 2173 | 203 |
| raw | bad_core_nearboundary | 0.823529 | 0.000000 | 0.000000 | 0.823529 | 0 | 0 | 21 |
| raw | bad_outlier_stress | 0.006849 | 0.000000 | 0.000000 | 0.006849 | 0 | 0 | 118 |
| badcal | original_test_all_10s+ | 0.775038 | 0.912637 | 0.708766 | 0.270073 | 311 | 1238 | 128 |
| badcal | original_all_10s+ | 0.840242 | 0.845039 | 0.787354 | 0.931126 | 2630 | 2173 | 191 |
| badcal | bad_core_nearboundary | 0.890756 | 0.000000 | 0.000000 | 0.890756 | 0 | 0 | 13 |
| badcal | bad_outlier_stress | 0.017123 | 0.000000 | 0.000000 | 0.017123 | 0 | 0 | 115 |

## Error Counts

- test errors raw: 1888
- bad outlier errors raw: 290
- bad core errors raw: 21
- good->medium raw: 316
- medium->good raw: 1238
- nonbad->bad raw: 23

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/predtop20_sqiquery_subject111_medium_gapheavy_mguard/original_error_waveform_panels.png)
