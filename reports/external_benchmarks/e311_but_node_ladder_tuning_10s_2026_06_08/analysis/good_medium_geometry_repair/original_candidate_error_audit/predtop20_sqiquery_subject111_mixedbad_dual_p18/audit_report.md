# Original Candidate Error Audit: predtop20_sqiquery_subject111_mixedbad_dual_p18

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.777516 | 0.898077 | 0.721871 | 0.309002 | 356 | 1125 | 53 |
| raw | original_all_10s+ | 0.828620 | 0.787361 | 0.841927 | 0.934910 | 3602 | 1529 | 112 |
| raw | bad_core_nearboundary | 0.974790 | 0.000000 | 0.000000 | 0.974790 | 0 | 0 | 3 |
| raw | bad_outlier_stress | 0.037671 | 0.000000 | 0.000000 | 0.037671 | 0 | 0 | 50 |
| badcal | original_test_all_10s+ | 0.762416 | 0.894505 | 0.693854 | 0.330900 | 350 | 1119 | 45 |
| badcal | original_all_10s+ | 0.818940 | 0.786423 | 0.812100 | 0.937559 | 3588 | 1521 | 99 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.058219 | 0.000000 | 0.000000 | 0.058219 | 0 | 0 | 45 |

## Error Counts

- test errors raw: 1886
- bad outlier errors raw: 281
- bad core errors raw: 3
- good->medium raw: 356
- medium->good raw: 1125
- nonbad->bad raw: 121

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/predtop20_sqiquery_subject111_mixedbad_dual_p18/original_error_waveform_panels.png)
