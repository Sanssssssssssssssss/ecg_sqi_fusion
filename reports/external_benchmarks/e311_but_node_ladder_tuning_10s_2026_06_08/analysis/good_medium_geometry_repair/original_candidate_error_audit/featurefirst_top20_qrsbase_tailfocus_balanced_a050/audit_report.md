# Original Candidate Error Audit: featurefirst_top20_qrsbase_tailfocus_balanced_a050

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.820809 | 0.915659 | 0.791685 | 0.294404 | 304 | 896 | 67 |
| raw | original_all_10s+ | 0.861512 | 0.834712 | 0.869213 | 0.932450 | 2813 | 1358 | 132 |
| raw | bad_core_nearboundary | 0.991597 | 0.000000 | 0.000000 | 0.991597 | 0 | 0 | 1 |
| raw | bad_outlier_stress | 0.010274 | 0.000000 | 0.000000 | 0.010274 | 0 | 0 | 66 |
| badcal | original_test_all_10s+ | 0.826000 | 0.915659 | 0.783778 | 0.486618 | 301 | 835 | 47 |
| badcal | original_all_10s+ | 0.862969 | 0.834712 | 0.865450 | 0.949101 | 2807 | 1297 | 104 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.277397 | 0.000000 | 0.000000 | 0.277397 | 0 | 0 | 47 |

## Error Counts

- test errors raw: 1519
- bad outlier errors raw: 289
- bad core errors raw: 1
- good->medium raw: 304
- medium->good raw: 896
- nonbad->bad raw: 29

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featurefirst_top20_qrsbase_tailfocus_balanced_a050/original_error_waveform_panels.png)
