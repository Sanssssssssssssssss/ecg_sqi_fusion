# Original Candidate Error Audit: predtop20_sqiquery_subject111_burstdrop_p16_specific

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.581928 | 0.439560 | 0.695436 | 0.620438 | 908 | 42 | 94 |
| raw | original_all_10s+ | 0.615275 | 0.369595 | 0.838069 | 0.959508 | 8827 | 84 | 152 |
| raw | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| raw | bad_outlier_stress | 0.465753 | 0.000000 | 0.000000 | 0.465753 | 0 | 0 | 94 |
| badcal | original_test_all_10s+ | 0.578625 | 0.439560 | 0.688884 | 0.622871 | 906 | 42 | 93 |
| badcal | original_all_10s+ | 0.613060 | 0.369595 | 0.830824 | 0.960265 | 8818 | 84 | 148 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.469178 | 0.000000 | 0.000000 | 0.469178 | 0 | 0 | 93 |

## Error Counts

- test errors raw: 3544
- bad outlier errors raw: 156
- bad core errors raw: 0
- good->medium raw: 908
- medium->good raw: 42
- nonbad->bad raw: 2438

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/predtop20_sqiquery_subject111_burstdrop_p16_specific/original_error_waveform_panels.png)
