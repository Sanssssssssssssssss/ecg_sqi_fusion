# Original Candidate Error Audit: featurefirst_top20_rawbeat_artifact_auxctx_jointartifact_badrecall_a050

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.731037 | 0.623901 | 0.816539 | 0.759124 | 949 | 108 | 61 |
| raw | original_all_10s+ | 0.684064 | 0.452913 | 0.912119 | 0.970861 | 8742 | 133 | 115 |
| raw | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| raw | bad_outlier_stress | 0.660959 | 0.000000 | 0.000000 | 0.660959 | 0 | 0 | 61 |
| badcal | original_test_all_10s+ | 0.727026 | 0.618956 | 0.812698 | 0.761557 | 946 | 107 | 60 |
| badcal | original_all_10s+ | 0.682971 | 0.451857 | 0.910237 | 0.971239 | 8733 | 132 | 113 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.664384 | 0.000000 | 0.000000 | 0.664384 | 0 | 0 | 60 |

## Error Counts

- test errors raw: 2280
- bad outlier errors raw: 99
- bad core errors raw: 0
- good->medium raw: 949
- medium->good raw: 108
- nonbad->bad raw: 1124

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featurefirst_top20_rawbeat_artifact_auxctx_jointartifact_badrecall_a050/original_error_waveform_panels.png)
