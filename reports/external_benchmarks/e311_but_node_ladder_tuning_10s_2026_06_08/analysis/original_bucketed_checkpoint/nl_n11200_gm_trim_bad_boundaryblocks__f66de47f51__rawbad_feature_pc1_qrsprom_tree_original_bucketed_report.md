# Original Bucketed Checkpoint Report

Report-only evaluation. It is not used for Clean/SemiClean/node selection.

## Checkpoint

- Variant: `nl_n11200_gm_trim_bad_boundaryblocks_n10000shell_thinprob_84079be0011f`
- Prediction mode: `rawbad_feature_pc1_qrsprom_tree`

## Buckets

- `original_all_10s+`: n=32956, acc=0.8344, macro-F1=0.8569, recall good/medium/bad=0.7350/0.9445/0.9336
- `original_test_all_10s+`: n=8477, acc=0.8351, macro-F1=0.6912, recall good/medium/bad=0.7810/0.9288/0.3041
- `original_test_good_medium_only`: n=8066, acc=0.8621, macro-F1=0.5787, recall good/medium/bad=0.7810/0.9288/0.0000
- `original_test_bad_core_near_boundary`: n=119, acc=1.0000, macro-F1=0.3333, recall good/medium/bad=0.0000/0.0000/1.0000
- `original_test_bad_outlier_stress`: n=292, acc=0.0205, macro-F1=0.0134, recall good/medium/bad=0.0000/0.0000/0.0205
- `original_test_drop_bad_outlier_reference`: n=8185, acc=0.8641, macro-F1=0.7831, recall good/medium/bad=0.7810/0.9288/1.0000
- `original_test_good_medium_overlap`: n=7492, acc=0.8516, macro-F1=0.5730, recall good/medium/bad=0.7787/0.9190/0.0000
- `original_all_bad_core_near_boundary`: n=4084, acc=1.0000, macro-F1=0.3333, recall good/medium/bad=0.0000/0.0000/1.0000
- `original_all_bad_outlier_stress`: n=1201, acc=0.7077, macro-F1=0.2763, recall good/medium/bad=0.0000/0.0000/0.7077

## Counts

- Original all 10s+: `32956` windows.
- Original test 10s+: `8477` windows.
- Bad outlier stress is reported separately because dropping it removes most original-test bad windows.

![bucketed confusions](nl_n11200_gm_trim_bad_boundaryblocks__f66de47f51__rawbad_feature_pc1_qrsprom_tree_original_bucketed_confusions.png)
