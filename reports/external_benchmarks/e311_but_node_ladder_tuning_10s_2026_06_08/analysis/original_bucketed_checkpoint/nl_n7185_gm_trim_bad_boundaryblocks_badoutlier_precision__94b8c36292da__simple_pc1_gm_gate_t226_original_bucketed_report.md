# Original Bucketed Checkpoint Report

Report-only evaluation. It is not used for Clean/SemiClean/node selection.

## Checkpoint

- Variant: `nl_n7185_gm_trim_bad_boundaryblocks_badoutlier_precision__94b8c36292da`
- Prediction mode: `simple_pc1_gm_gate_t226`

## Buckets

- `original_all_10s+`: n=32956, acc=0.8306, macro-F1=0.8427, recall good/medium/bad=0.8107/0.8100/0.9362
- `original_test_all_10s+`: n=8477, acc=0.7581, macro-F1=0.6160, recall good/medium/bad=0.9253/0.6602/0.3309
- `original_test_good_medium_only`: n=8066, acc=0.7798, macro-F1=0.5341, recall good/medium/bad=0.9253/0.6602/0.0000
- `original_test_bad_core_near_boundary`: n=119, acc=0.9328, macro-F1=0.3217, recall good/medium/bad=0.0000/0.0000/0.9328
- `original_test_bad_outlier_stress`: n=292, acc=0.0856, macro-F1=0.0526, recall good/medium/bad=0.0000/0.0000/0.0856
- `original_test_drop_bad_outlier_reference`: n=8185, acc=0.7820, macro-F1=0.6429, recall good/medium/bad=0.9253/0.6602/0.9328
- `original_test_good_medium_overlap`: n=7492, acc=0.7631, macro-F1=0.5210, recall good/medium/bad=0.9245/0.6136/0.0000
- `original_all_bad_core_near_boundary`: n=4084, acc=0.9978, macro-F1=0.3330, recall good/medium/bad=0.0000/0.0000/0.9978
- `original_all_bad_outlier_stress`: n=1201, acc=0.7269, macro-F1=0.2806, recall good/medium/bad=0.0000/0.0000/0.7269

## Counts

- Original all 10s+: `32956` windows.
- Original test 10s+: `8477` windows.
- Bad outlier stress is reported separately because dropping it removes most original-test bad windows.

![bucketed confusions](nl_n7185_gm_trim_bad_boundaryblocks_badoutlier_precision__94b8c36292da__simple_pc1_gm_gate_t226_original_bucketed_confusions.png)
