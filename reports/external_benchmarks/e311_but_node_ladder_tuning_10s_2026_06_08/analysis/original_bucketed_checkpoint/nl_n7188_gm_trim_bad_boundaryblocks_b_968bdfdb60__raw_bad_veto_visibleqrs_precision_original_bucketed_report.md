# Original Bucketed Checkpoint Report

Report-only evaluation. It is not used for Clean/SemiClean/node selection.

## Checkpoint

- Variant: `nl_n7188_gm_trim_bad_boundaryblocks_badoutlier_visqrsnarr_a364001dc6cf`
- Prediction mode: `raw_bad_veto_visibleqrs_precision`

## Buckets

- `original_all_10s+`: n=32956, acc=0.7934, macro-F1=0.8232, recall good/medium/bad=0.6778/0.9009/0.9499
- `original_test_all_10s+`: n=8477, acc=0.8017, macro-F1=0.7086, recall good/medium/bad=0.7516/0.8701/0.5085
- `original_test_good_medium_only`: n=8066, acc=0.8166, macro-F1=0.5508, recall good/medium/bad=0.7516/0.8701/0.0000
- `original_test_bad_core_near_boundary`: n=119, acc=1.0000, macro-F1=0.3333, recall good/medium/bad=0.0000/0.0000/1.0000
- `original_test_bad_outlier_stress`: n=292, acc=0.3082, macro-F1=0.1571, recall good/medium/bad=0.0000/0.0000/0.3082
- `original_test_drop_bad_outlier_reference`: n=8185, acc=0.8193, macro-F1=0.7207, recall good/medium/bad=0.7516/0.8701/1.0000
- `original_test_good_medium_overlap`: n=7492, acc=0.8047, macro-F1=0.5439, recall good/medium/bad=0.7490/0.8563/0.0000
- `original_all_bad_core_near_boundary`: n=4084, acc=0.9998, macro-F1=0.3333, recall good/medium/bad=0.0000/0.0000/0.9998
- `original_all_bad_outlier_stress`: n=1201, acc=0.7802, macro-F1=0.2922, recall good/medium/bad=0.0000/0.0000/0.7802

## Counts

- Original all 10s+: `32956` windows.
- Original test 10s+: `8477` windows.
- Bad outlier stress is reported separately because dropping it removes most original-test bad windows.

![bucketed confusions](nl_n7188_gm_trim_bad_boundaryblocks_b_968bdfdb60__raw_bad_veto_visibleqrs_precision_original_bucketed_confusions.png)
