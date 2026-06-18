# Original Candidate Error Audit: featurefirst_top20_qrsbase_dualcoreout_physioctx_direct_balanced_a050

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.838976 | 0.908242 | 0.802756 | 0.615572 | 330 | 704 | 54 |
| raw | original_all_10s+ | 0.867824 | 0.841401 | 0.865450 | 0.957805 | 2699 | 1253 | 118 |
| raw | bad_core_nearboundary | 0.932773 | 0.000000 | 0.000000 | 0.932773 | 0 | 0 | 8 |
| raw | bad_outlier_stress | 0.486301 | 0.000000 | 0.000000 | 0.486301 | 0 | 0 | 46 |
| badcal | original_test_all_10s+ | 0.839920 | 0.908242 | 0.801853 | 0.644769 | 330 | 701 | 52 |
| badcal | original_all_10s+ | 0.868097 | 0.841401 | 0.865073 | 0.960265 | 2699 | 1250 | 115 |
| badcal | bad_core_nearboundary | 0.949580 | 0.000000 | 0.000000 | 0.949580 | 0 | 0 | 6 |
| badcal | bad_outlier_stress | 0.520548 | 0.000000 | 0.000000 | 0.520548 | 0 | 0 | 46 |

## Error Counts

- test errors raw: 1365
- bad outlier errors raw: 150
- bad core errors raw: 8
- good->medium raw: 330
- medium->good raw: 704
- nonbad->bad raw: 173

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featurefirst_top20_qrsbase_dualcoreout_physioctx_direct_balanced_a050/original_error_waveform_panels.png)
