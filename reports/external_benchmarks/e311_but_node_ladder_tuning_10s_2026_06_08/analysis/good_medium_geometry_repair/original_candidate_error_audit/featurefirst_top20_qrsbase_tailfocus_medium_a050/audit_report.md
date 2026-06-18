# Original Candidate Error Audit: featurefirst_top20_qrsbase_tailfocus_medium_a050

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.828831 | 0.915934 | 0.807049 | 0.291971 | 302 | 827 | 77 |
| raw | original_all_10s+ | 0.865002 | 0.840521 | 0.870719 | 0.932450 | 2713 | 1340 | 141 |
| raw | bad_core_nearboundary | 0.983193 | 0.000000 | 0.000000 | 0.983193 | 0 | 0 | 2 |
| raw | bad_outlier_stress | 0.010274 | 0.000000 | 0.000000 | 0.010274 | 0 | 0 | 75 |
| badcal | original_test_all_10s+ | 0.831898 | 0.915934 | 0.800271 | 0.428224 | 300 | 789 | 59 |
| badcal | original_all_10s+ | 0.865851 | 0.840521 | 0.867426 | 0.944371 | 2710 | 1302 | 117 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.195205 | 0.000000 | 0.000000 | 0.195205 | 0 | 0 | 59 |

## Error Counts

- test errors raw: 1451
- bad outlier errors raw: 289
- bad core errors raw: 2
- good->medium raw: 302
- medium->good raw: 827
- nonbad->bad raw: 31

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featurefirst_top20_qrsbase_tailfocus_medium_a050/original_error_waveform_panels.png)
