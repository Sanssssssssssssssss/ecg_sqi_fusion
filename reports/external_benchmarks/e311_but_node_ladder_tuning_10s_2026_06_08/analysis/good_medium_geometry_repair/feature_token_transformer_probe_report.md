# Feature-Token Transformer Probe

This is a feature-assisted Transformer over selected SQI/geometry columns. It is not waveform-only.
Selection uses the current BUT/node train/val split; the BUT test split is held out for reporting.

## Main Metrics

| candidate | bucket | n | acc | macro_f1 | good_recall | medium_recall | bad_recall | good_to_medium | medium_to_good | bad_to_medium |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| featuretx_top14 | node_test | 8477 | 0.878849 | 0.773303 | 0.990385 | 0.833710 | 0.377129 | 35 | 726 | 152 |
| featuretx_top14 | original_test_all_10s+ | 8477 | 0.878849 | 0.773303 | 0.990385 | 0.833710 | 0.377129 | 35 | 726 | 152 |
| featuretx_top14 | original_all_10s+ | 32956 | 0.964862 | 0.964877 | 0.992607 | 0.927362 | 0.950804 | 126 | 762 | 154 |
| featuretx_top20 | node_test | 8477 | 0.908458 | 0.932228 | 0.999725 | 0.825124 | 0.997567 | 0 | 766 | 0 |
| featuretx_top20 | original_test_all_10s+ | 8477 | 0.908458 | 0.932228 | 0.999725 | 0.825124 | 0.997567 | 0 | 766 | 0 |
| featuretx_top20 | original_all_10s+ | 32956 | 0.965864 | 0.970558 | 0.996127 | 0.900546 | 0.999622 | 65 | 1048 | 0 |
| featuretx_top22 | node_test | 8477 | 0.910935 | 0.909913 | 0.996429 | 0.846588 | 0.846715 | 13 | 671 | 63 |
| featuretx_top22 | original_test_all_10s+ | 8477 | 0.910935 | 0.909913 | 0.996429 | 0.846588 | 0.846715 | 13 | 671 | 63 |
| featuretx_top22 | original_all_10s+ | 32956 | 0.967108 | 0.970435 | 0.993546 | 0.914283 | 0.988079 | 110 | 902 | 63 |

## All Buckets

| candidate | bucket | n | acc | macro_f1 | good_recall | medium_recall | bad_recall | good_to_medium | medium_to_good | bad_to_medium |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| featuretx_top14 | node_test | 8477 | 0.878849 | 0.773303 | 0.990385 | 0.833710 | 0.377129 | 35 | 726 | 152 |
| featuretx_top14 | original_test_all_10s+ | 8477 | 0.878849 | 0.773303 | 0.990385 | 0.833710 | 0.377129 | 35 | 726 | 152 |
| featuretx_top14 | original_all_10s+ | 32956 | 0.964862 | 0.964877 | 0.992607 | 0.927362 | 0.950804 | 126 | 762 | 154 |
| featuretx_top14 | bad_core_nearboundary | 4084 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| featuretx_top14 | bad_outlier_stress | 1201 | 0.783514 | 0.292873 | 0.000000 | 0.000000 | 0.783514 | 0 | 0 | 154 |
| featuretx_top20 | node_test | 8477 | 0.908458 | 0.932228 | 0.999725 | 0.825124 | 0.997567 | 0 | 766 | 0 |
| featuretx_top20 | original_test_all_10s+ | 8477 | 0.908458 | 0.932228 | 0.999725 | 0.825124 | 0.997567 | 0 | 766 | 0 |
| featuretx_top20 | original_all_10s+ | 32956 | 0.965864 | 0.970558 | 0.996127 | 0.900546 | 0.999622 | 65 | 1048 | 0 |
| featuretx_top20 | bad_core_nearboundary | 4084 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| featuretx_top20 | bad_outlier_stress | 1201 | 0.998335 | 0.499583 | 0.000000 | 0.000000 | 0.998335 | 0 | 0 | 0 |
| featuretx_top22 | node_test | 8477 | 0.910935 | 0.909913 | 0.996429 | 0.846588 | 0.846715 | 13 | 671 | 63 |
| featuretx_top22 | original_test_all_10s+ | 8477 | 0.910935 | 0.909913 | 0.996429 | 0.846588 | 0.846715 | 13 | 671 | 63 |
| featuretx_top22 | original_all_10s+ | 32956 | 0.967108 | 0.970435 | 0.993546 | 0.914283 | 0.988079 | 110 | 902 | 63 |
| featuretx_top22 | bad_core_nearboundary | 4084 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0 | 0 | 0 |
| featuretx_top22 | bad_outlier_stress | 1201 | 0.947544 | 0.486533 | 0.000000 | 0.000000 | 0.947544 | 0 | 0 | 63 |

Metrics CSV: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\feature_token_transformer_probe_metrics.csv`