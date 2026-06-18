# Original Bucketed Checkpoint Report

Report-only evaluation. It is not used for Clean/SemiClean/node selection.

## Checkpoint

- Variant: `nl_n7179_gm_trim_bad_boundaryblocks_ultramicro_goodmed_n7_7a210e9eef05`
- Prediction mode: `simple_pc1_gm_gate_t254`

## Buckets

- `original_all_10s+`: n=32956, acc=0.8208, macro-F1=0.8472, recall good/medium/bad=0.7747/0.8425/0.9256
- `original_test_all_10s+`: n=8477, acc=0.7714, macro-F1=0.6382, recall good/medium/bad=0.9190/0.7022/0.2092
- `original_test_good_medium_only`: n=8066, acc=0.8000, macro-F1=0.5332, recall good/medium/bad=0.9190/0.7022/0.0000
- `original_test_bad_core_near_boundary`: n=119, acc=0.7227, macro-F1=0.2797, recall good/medium/bad=0.0000/0.0000/0.7227
- `original_test_bad_outlier_stress`: n=292, acc=0.0000, macro-F1=0.0000, recall good/medium/bad=0.0000/0.0000/0.0000
- `original_test_drop_bad_outlier_reference`: n=8185, acc=0.7989, macro-F1=0.8118, recall good/medium/bad=0.9190/0.7022/0.7227
- `original_test_good_medium_overlap`: n=7492, acc=0.7847, macro-F1=0.5217, recall good/medium/bad=0.9181/0.6612/0.0000
- `original_all_bad_core_near_boundary`: n=4084, acc=0.9919, macro-F1=0.3320, recall good/medium/bad=0.0000/0.0000/0.9919
- `original_all_bad_outlier_stress`: n=1201, acc=0.7002, macro-F1=0.2746, recall good/medium/bad=0.0000/0.0000/0.7002

## Counts

- Original all 10s+: `32956` windows.
- Original test 10s+: `8477` windows.
- Bad outlier stress is reported separately because dropping it removes most original-test bad windows.

![bucketed confusions](nl_n7179_gm_trim_bad_boundaryblocks_ultramicro_goodmed_n7_7a210e9eef05__simple_pc1_gm_gate_t254_original_bucketed_confusions.png)
