# Original Candidate Error Audit: featurefirst_top20_hardrec_badlite_a050

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.798042 | 0.911538 | 0.748531 | 0.326034 | 312 | 966 | 50 |
| raw | original_all_10s+ | 0.852591 | 0.833891 | 0.841551 | 0.935099 | 2802 | 1497 | 114 |
| raw | bad_core_nearboundary | 0.932773 | 0.000000 | 0.000000 | 0.932773 | 0 | 0 | 8 |
| raw | bad_outlier_stress | 0.078767 | 0.000000 | 0.000000 | 0.078767 | 0 | 0 | 42 |
| badcal | original_test_all_10s+ | 0.756400 | 0.893681 | 0.673294 | 0.435523 | 290 | 604 | 36 |
| badcal | original_all_10s+ | 0.834142 | 0.820748 | 0.800527 | 0.944939 | 2752 | 1105 | 93 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.205479 | 0.000000 | 0.000000 | 0.205479 | 0 | 0 | 36 |

## Error Counts

- test errors raw: 1712
- bad outlier errors raw: 269
- bad core errors raw: 8
- good->medium raw: 312
- medium->good raw: 966
- nonbad->bad raw: 157

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featurefirst_top20_hardrec_badlite_a050/original_error_waveform_panels.png)
