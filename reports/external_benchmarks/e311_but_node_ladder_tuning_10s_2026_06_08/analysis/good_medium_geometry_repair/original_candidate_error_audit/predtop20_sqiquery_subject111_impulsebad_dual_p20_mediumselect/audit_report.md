# Original Candidate Error Audit: predtop20_sqiquery_subject111_impulsebad_dual_p20_mediumselect

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.797924 | 0.892857 | 0.766155 | 0.299270 | 380 | 973 | 81 |
| raw | original_all_10s+ | 0.820063 | 0.749399 | 0.876553 | 0.934342 | 4252 | 1230 | 139 |
| raw | bad_core_nearboundary | 0.991597 | 0.000000 | 0.000000 | 0.991597 | 0 | 0 | 1 |
| raw | bad_outlier_stress | 0.017123 | 0.000000 | 0.000000 | 0.017123 | 0 | 0 | 80 |
| badcal | original_test_all_10s+ | 0.790138 | 0.890110 | 0.752146 | 0.313869 | 378 | 960 | 76 |
| badcal | original_all_10s+ | 0.816786 | 0.748577 | 0.867049 | 0.935667 | 4239 | 1212 | 133 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.034247 | 0.000000 | 0.000000 | 0.034247 | 0 | 0 | 76 |

## Error Counts

- test errors raw: 1713
- bad outlier errors raw: 287
- bad core errors raw: 1
- good->medium raw: 380
- medium->good raw: 973
- nonbad->bad raw: 72

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/predtop20_sqiquery_subject111_impulsebad_dual_p20_mediumselect/original_error_waveform_panels.png)
