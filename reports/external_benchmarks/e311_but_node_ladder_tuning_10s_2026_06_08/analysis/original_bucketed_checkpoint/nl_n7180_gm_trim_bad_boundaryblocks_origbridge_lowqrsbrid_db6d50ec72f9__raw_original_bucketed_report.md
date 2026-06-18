# Original Bucketed Checkpoint Report

Report-only evaluation. It is not used for Clean/SemiClean/node selection.

## Checkpoint

- Variant: `nl_n7180_gm_trim_bad_boundaryblocks_origbridge_lowqrsbrid_db6d50ec72f9`
- Prediction mode: `raw`

## Buckets

- `original_all_10s+`: n=32956, acc=0.7581, macro-F1=0.7979, recall good/medium/bad=0.6215/0.9015/0.9099
- `original_test_all_10s+`: n=8477, acc=0.7557, macro-F1=0.5209, recall good/medium/bad=0.7497/0.8301/0.0073
- `original_test_good_medium_only`: n=8066, acc=0.7938, macro-F1=0.5290, recall good/medium/bad=0.7497/0.8301/0.0000
- `original_test_bad_core_near_boundary`: n=119, acc=0.0000, macro-F1=0.0000, recall good/medium/bad=0.0000/0.0000/0.0000
- `original_test_bad_outlier_stress`: n=292, acc=0.0103, macro-F1=0.0068, recall good/medium/bad=0.0000/0.0000/0.0103
- `original_test_drop_bad_outlier_reference`: n=8185, acc=0.7823, macro-F1=0.5254, recall good/medium/bad=0.7497/0.8301/0.0000
- `original_test_good_medium_overlap`: n=7492, acc=0.7787, macro-F1=0.5203, recall good/medium/bad=0.7471/0.8080/0.0000
- `original_all_bad_core_near_boundary`: n=4084, acc=0.9709, macro-F1=0.3284, recall good/medium/bad=0.0000/0.0000/0.9709
- `original_all_bad_outlier_stress`: n=1201, acc=0.7027, macro-F1=0.2751, recall good/medium/bad=0.0000/0.0000/0.7027

## Counts

- Original all 10s+: `32956` windows.
- Original test 10s+: `8477` windows.
- Bad outlier stress is reported separately because dropping it removes most original-test bad windows.

![bucketed confusions](nl_n7180_gm_trim_bad_boundaryblocks_origbridge_lowqrsbrid_db6d50ec72f9__raw_original_bucketed_confusions.png)
