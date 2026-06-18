# Original Candidate Error Audit: featurefirst_top20_qrsbase_primres_current_conservative_a050

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.830011 | 0.906593 | 0.788748 | 0.596107 | 333 | 739 | 40 |
| raw | original_all_10s+ | 0.867702 | 0.848501 | 0.854159 | 0.956859 | 2569 | 1332 | 101 |
| raw | bad_core_nearboundary | 0.991597 | 0.000000 | 0.000000 | 0.991597 | 0 | 0 | 1 |
| raw | bad_outlier_stress | 0.434932 | 0.000000 | 0.000000 | 0.434932 | 0 | 0 | 39 |
| badcal | original_test_all_10s+ | 0.831190 | 0.906593 | 0.787845 | 0.630170 | 333 | 732 | 40 |
| badcal | original_all_10s+ | 0.868006 | 0.848501 | 0.853688 | 0.959697 | 2569 | 1325 | 100 |
| badcal | bad_core_nearboundary | 0.991597 | 0.000000 | 0.000000 | 0.991597 | 0 | 0 | 1 |
| badcal | bad_outlier_stress | 0.482877 | 0.000000 | 0.000000 | 0.482877 | 0 | 0 | 39 |

## Error Counts

- test errors raw: 1441
- bad outlier errors raw: 165
- bad core errors raw: 1
- good->medium raw: 333
- medium->good raw: 739
- nonbad->bad raw: 203

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featurefirst_top20_qrsbase_primres_current_conservative_a050/original_error_waveform_panels.png)
