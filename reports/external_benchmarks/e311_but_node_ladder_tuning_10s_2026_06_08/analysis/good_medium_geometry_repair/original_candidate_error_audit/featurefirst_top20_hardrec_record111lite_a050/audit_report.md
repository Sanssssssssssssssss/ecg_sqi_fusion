# Original Candidate Error Audit: featurefirst_top20_hardrec_record111lite_a050

Original BUT is report-only; no training or selection uses these rows.

## Buckets

| mode | bucket | acc | good R | medium R | bad R | g->m | m->g | b->m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raw | original_test_all_10s+ | 0.798632 | 0.869231 | 0.782874 | 0.343066 | 435 | 761 | 56 |
| raw | original_all_10s+ | 0.803920 | 0.712727 | 0.883892 | 0.937181 | 4806 | 971 | 116 |
| raw | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| raw | bad_outlier_stress | 0.075342 | 0.000000 | 0.000000 | 0.075342 | 0 | 0 | 56 |
| badcal | original_test_all_10s+ | 0.778931 | 0.852473 | 0.754180 | 0.394161 | 424 | 630 | 50 |
| badcal | original_all_10s+ | 0.791844 | 0.701578 | 0.861874 | 0.942100 | 4769 | 820 | 105 |
| badcal | bad_core_nearboundary | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| badcal | bad_outlier_stress | 0.147260 | 0.000000 | 0.000000 | 0.147260 | 0 | 0 | 50 |

## Error Counts

- test errors raw: 1707
- bad outlier errors raw: 270
- bad core errors raw: 0
- good->medium raw: 435
- medium->good raw: 761
- nonbad->bad raw: 241

![error panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/original_candidate_error_audit/featurefirst_top20_hardrec_record111lite_a050/original_error_waveform_panels.png)
