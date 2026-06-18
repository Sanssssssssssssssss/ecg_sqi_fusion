# Original Bucketed Checkpoint Report

Report-only evaluation. It is not used for Clean/SemiClean/node selection.

## Checkpoint

- Variant: `nl_n7185_gm_trim_bad_boundaryblocks_badoutlier_precision__94b8c36292da`
- Prediction mode: `raw`

## Buckets

- `original_all_10s+`: n=32956, acc=0.8141, macro-F1=0.8173, recall good/medium/bad=0.9027/0.6114/0.9362
- `original_test_all_10s+`: n=8477, acc=0.7155, macro-F1=0.5852, recall good/medium/bad=0.8989/0.6003/0.3309
- `original_test_good_medium_only`: n=8066, acc=0.7351, macro-F1=0.5027, recall good/medium/bad=0.8989/0.6003/0.0000
- `original_test_bad_core_near_boundary`: n=119, acc=0.9328, macro-F1=0.3217, recall good/medium/bad=0.0000/0.0000/0.9328
- `original_test_bad_outlier_stress`: n=292, acc=0.0856, macro-F1=0.0526, recall good/medium/bad=0.0000/0.0000/0.0856
- `original_test_drop_bad_outlier_reference`: n=8185, acc=0.7379, macro-F1=0.6116, recall good/medium/bad=0.8989/0.6003/0.9328
- `original_test_good_medium_overlap`: n=7492, acc=0.7221, macro-F1=0.4916, recall good/medium/bad=0.8978/0.5594/0.0000
- `original_all_bad_core_near_boundary`: n=4084, acc=0.9978, macro-F1=0.3330, recall good/medium/bad=0.0000/0.0000/0.9978
- `original_all_bad_outlier_stress`: n=1201, acc=0.7269, macro-F1=0.2806, recall good/medium/bad=0.0000/0.0000/0.7269

## Counts

- Original all 10s+: `32956` windows.
- Original test 10s+: `8477` windows.
- Bad outlier stress is reported separately because dropping it removes most original-test bad windows.

![bucketed confusions](nl_n7185_gm_trim_bad_boundaryblocks_badoutlier_precision__94b8c36292da__raw_original_bucketed_confusions.png)
