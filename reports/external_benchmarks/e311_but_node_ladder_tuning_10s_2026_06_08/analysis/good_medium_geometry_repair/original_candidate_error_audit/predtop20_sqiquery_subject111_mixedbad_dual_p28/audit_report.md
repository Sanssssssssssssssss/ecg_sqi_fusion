# Original Candidate Error Audit: predtop20_sqiquery_subject111_mixedbad_dual_p28

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.796980 | 0.905769 | 0.753954 | 0.296837 | 340 | 1047 | 56 |
| raw | original_all_10s+ | 0.852227 | 0.833069 | 0.842868 | 0.932829 | 2840 | 1615 | 119 |
| raw | bad_core_nearboundary | 0.966387 | 0.000000 | 0.000000 | 0.966387 | 0 | 0 | 4 |
| raw | bad_outlier_stress | 0.023973 | 0.000000 | 0.000000 | 0.023973 | 0 | 0 | 52 |
| badcal | original_test_all_10s+ | 0.793323 | 0.903846 | 0.747402 | 0.309002 | 338 | 1030 | 53 |
| badcal | original_all_10s+ | 0.850528 | 0.832424 | 0.837693 | 0.934721 | 2838 | 1598 | 112 |
| badcal | bad_core_nearboundary | 0.991597 | 0.000000 | 0.000000 | 0.991597 | 0 | 0 | 1 |
| badcal | bad_outlier_stress | 0.030822 | 0.000000 | 0.000000 | 0.030822 | 0 | 0 | 52 |

## Error Counts

- test errors raw: 1721
- bad outlier errors raw: 285
- bad core errors raw: 4
- good->medium raw: 340
- medium->good raw: 1047
- nonbad->bad raw: 45

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/predtop20_sqiquery_subject111_mixedbad_dual_p28/original_error_waveform_panels.png)
