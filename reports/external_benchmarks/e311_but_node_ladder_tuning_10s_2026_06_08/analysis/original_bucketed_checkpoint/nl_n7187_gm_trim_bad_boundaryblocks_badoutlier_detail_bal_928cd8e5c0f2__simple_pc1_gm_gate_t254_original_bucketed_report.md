# Original Bucketed Checkpoint Report

Report-only evaluation. It is not used for Clean/SemiClean/node selection.

## Checkpoint

- Variant: `nl_n7187_gm_trim_bad_boundaryblocks_badoutlier_detail_bal_928cd8e5c0f2`
- Prediction mode: `simple_pc1_gm_gate_t254`

## Buckets

- `original_all_10s+`: n=32956, acc=0.8148, macro-F1=0.8299, recall good/medium/bad=0.7693/0.8265/0.9377
- `original_test_all_10s+`: n=8477, acc=0.7623, macro-F1=0.6184, recall good/medium/bad=0.9137/0.6767/0.3431
- `original_test_good_medium_only`: n=8066, acc=0.7837, macro-F1=0.5393, recall good/medium/bad=0.9137/0.6767/0.0000
- `original_test_bad_core_near_boundary`: n=119, acc=1.0000, macro-F1=0.3333, recall good/medium/bad=0.0000/0.0000/1.0000
- `original_test_bad_outlier_stress`: n=292, acc=0.0753, macro-F1=0.0467, recall good/medium/bad=0.0000/0.0000/0.0753
- `original_test_drop_bad_outlier_reference`: n=8185, acc=0.7868, macro-F1=0.6451, recall good/medium/bad=0.9137/0.6767/1.0000
- `original_test_good_medium_overlap`: n=7492, acc=0.7706, macro-F1=0.5284, recall good/medium/bad=0.9128/0.6388/0.0000
- `original_all_bad_core_near_boundary`: n=4084, acc=0.9998, macro-F1=0.3333, recall good/medium/bad=0.0000/0.0000/0.9998
- `original_all_bad_outlier_stress`: n=1201, acc=0.7269, macro-F1=0.2806, recall good/medium/bad=0.0000/0.0000/0.7269

## Counts

- Original all 10s+: `32956` windows.
- Original test 10s+: `8477` windows.
- Bad outlier stress is reported separately because dropping it removes most original-test bad windows.

![bucketed confusions](nl_n7187_gm_trim_bad_boundaryblocks_badoutlier_detail_bal_928cd8e5c0f2__simple_pc1_gm_gate_t254_original_bucketed_confusions.png)
