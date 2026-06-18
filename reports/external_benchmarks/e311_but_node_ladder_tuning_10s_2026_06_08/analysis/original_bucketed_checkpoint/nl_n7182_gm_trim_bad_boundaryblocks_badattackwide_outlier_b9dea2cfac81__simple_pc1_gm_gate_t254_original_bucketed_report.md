# Original Bucketed Checkpoint Report

Report-only evaluation. It is not used for Clean/SemiClean/node selection.

## Checkpoint

- Variant: `nl_n7182_gm_trim_bad_boundaryblocks_badattackwide_outlier_b9dea2cfac81`
- Prediction mode: `simple_pc1_gm_gate_t254`

## Buckets

- `original_all_10s+`: n=32956, acc=0.8145, macro-F1=0.8276, recall good/medium/bad=0.7706/0.8344/0.9164
- `original_test_all_10s+`: n=8477, acc=0.7559, macro-F1=0.5618, recall good/medium/bad=0.9140/0.6859/0.1095
- `original_test_good_medium_only`: n=8066, acc=0.7889, macro-F1=0.5449, recall good/medium/bad=0.9140/0.6859/0.0000
- `original_test_bad_core_near_boundary`: n=119, acc=0.0000, macro-F1=0.0000, recall good/medium/bad=0.0000/0.0000/0.0000
- `original_test_bad_outlier_stress`: n=292, acc=0.1541, macro-F1=0.0890, recall good/medium/bad=0.0000/0.0000/0.1541
- `original_test_drop_bad_outlier_reference`: n=8185, acc=0.7774, macro-F1=0.5409, recall good/medium/bad=0.9140/0.6859/0.0000
- `original_test_good_medium_overlap`: n=7492, acc=0.7727, macro-F1=0.5328, recall good/medium/bad=0.9131/0.6427/0.0000
- `original_all_bad_core_near_boundary`: n=4084, acc=0.9699, macro-F1=0.3282, recall good/medium/bad=0.0000/0.0000/0.9699
- `original_all_bad_outlier_stress`: n=1201, acc=0.7344, macro-F1=0.2823, recall good/medium/bad=0.0000/0.0000/0.7344

## Counts

- Original all 10s+: `32956` windows.
- Original test 10s+: `8477` windows.
- Bad outlier stress is reported separately because dropping it removes most original-test bad windows.

![bucketed confusions](nl_n7182_gm_trim_bad_boundaryblocks_badattackwide_outlier_b9dea2cfac81__simple_pc1_gm_gate_t254_original_bucketed_confusions.png)
