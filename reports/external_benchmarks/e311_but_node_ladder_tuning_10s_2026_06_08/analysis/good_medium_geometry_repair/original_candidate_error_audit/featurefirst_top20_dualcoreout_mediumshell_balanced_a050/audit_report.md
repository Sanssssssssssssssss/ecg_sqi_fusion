# Original Candidate Error Audit: featurefirst_top20_dualcoreout_mediumshell_balanced_a050

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.844520 | 0.904670 | 0.848848 | 0.265207 | 346 | 666 | 103 |
| raw | original_all_10s+ | 0.872588 | 0.859238 | 0.865732 | 0.929423 | 2398 | 1424 | 172 |
| raw | bad_core_nearboundary | 0.899160 | 0.000000 | 0.000000 | 0.899160 | 0 | 0 | 12 |
| raw | bad_outlier_stress | 0.006849 | 0.000000 | 0.000000 | 0.006849 | 0 | 0 | 91 |
| badcal | original_test_all_10s+ | 0.845228 | 0.904670 | 0.837325 | 0.403893 | 343 | 644 | 69 |
| badcal | original_all_10s+ | 0.872739 | 0.859238 | 0.860181 | 0.941533 | 2394 | 1402 | 132 |
| badcal | bad_core_nearboundary | 0.974790 | 0.000000 | 0.000000 | 0.974790 | 0 | 0 | 3 |
| badcal | bad_outlier_stress | 0.171233 | 0.000000 | 0.000000 | 0.171233 | 0 | 0 | 66 |

## Error Counts

- test errors raw: 1318
- bad outlier errors raw: 290
- bad core errors raw: 12
- good->medium raw: 346
- medium->good raw: 666
- nonbad->bad raw: 4

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featurefirst_top20_dualcoreout_mediumshell_balanced_a050/original_error_waveform_panels.png)
