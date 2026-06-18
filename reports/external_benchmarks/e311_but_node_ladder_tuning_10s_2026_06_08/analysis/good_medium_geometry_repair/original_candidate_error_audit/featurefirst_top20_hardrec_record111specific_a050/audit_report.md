# Original Candidate Error Audit: featurefirst_top20_hardrec_record111specific_a050

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.813731 | 0.877198 | 0.809535 | 0.296837 | 401 | 688 | 73 |
| raw | original_all_10s+ | 0.853502 | 0.815819 | 0.874953 | 0.931883 | 3049 | 1158 | 142 |
| raw | bad_core_nearboundary | 0.907563 | 0.000000 | 0.000000 | 0.907563 | 0 | 0 | 11 |
| raw | bad_outlier_stress | 0.047945 | 0.000000 | 0.000000 | 0.047945 | 0 | 0 | 62 |
| badcal | original_test_all_10s+ | 0.774566 | 0.837088 | 0.753276 | 0.450122 | 372 | 486 | 48 |
| badcal | original_all_10s+ | 0.837359 | 0.797512 | 0.847290 | 0.945885 | 2951 | 928 | 107 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.226027 | 0.000000 | 0.000000 | 0.226027 | 0 | 0 | 48 |

## Error Counts

- test errors raw: 1579
- bad outlier errors raw: 278
- bad core errors raw: 11
- good->medium raw: 401
- medium->good raw: 688
- nonbad->bad raw: 201

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featurefirst_top20_hardrec_record111specific_a050/original_error_waveform_panels.png)
