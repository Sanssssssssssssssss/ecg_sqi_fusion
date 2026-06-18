# Original Candidate Error Audit: predtop20_sqiquery_subject111_shift_stress_originalcurve

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.772797 | 0.902198 | 0.704700 | 0.360097 | 328 | 1106 | 42 |
| raw | original_all_10s+ | 0.835781 | 0.815995 | 0.817557 | 0.936235 | 3027 | 1672 | 106 |
| raw | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| raw | bad_outlier_stress | 0.099315 | 0.000000 | 0.000000 | 0.099315 | 0 | 0 | 42 |
| badcal | original_test_all_10s+ | 0.754748 | 0.885165 | 0.679169 | 0.413625 | 319 | 966 | 32 |
| badcal | original_all_10s+ | 0.826830 | 0.807076 | 0.800715 | 0.943046 | 2991 | 1510 | 90 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.174658 | 0.000000 | 0.000000 | 0.174658 | 0 | 0 | 32 |

## Error Counts

- test errors raw: 1926
- bad outlier errors raw: 263
- bad core errors raw: 0
- good->medium raw: 328
- medium->good raw: 1106
- nonbad->bad raw: 229

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/predtop20_sqiquery_subject111_shift_stress_originalcurve/original_error_waveform_panels.png)
