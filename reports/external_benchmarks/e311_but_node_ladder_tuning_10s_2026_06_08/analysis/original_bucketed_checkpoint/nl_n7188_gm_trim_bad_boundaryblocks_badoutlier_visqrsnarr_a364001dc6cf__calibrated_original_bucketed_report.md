# Original Bucketed Checkpoint Report

Report-only evaluation. It is not used for Clean/SemiClean/node selection.

## Checkpoint

- Variant: `nl_n7188_gm_trim_bad_boundaryblocks_badoutlier_visqrsnarr_a364001dc6cf`
- Prediction mode: `calibrated`

## Buckets

- `original_all_10s+`: n=32956, acc=0.7926, macro-F1=0.8234, recall good/medium/bad=0.6785/0.9045/0.9353
- `original_test_all_10s+`: n=8477, acc=0.7966, macro-F1=0.6665, recall good/medium/bad=0.7533/0.8764/0.3212
- `original_test_good_medium_only`: n=8066, acc=0.8209, macro-F1=0.5513, recall good/medium/bad=0.7533/0.8764/0.0000
- `original_test_bad_core_near_boundary`: n=119, acc=1.0000, macro-F1=0.3333, recall good/medium/bad=0.0000/0.0000/1.0000
- `original_test_bad_outlier_stress`: n=292, acc=0.0445, macro-F1=0.0284, recall good/medium/bad=0.0000/0.0000/0.0445
- `original_test_drop_bad_outlier_reference`: n=8185, acc=0.8235, macro-F1=0.7472, recall good/medium/bad=0.7533/0.8764/1.0000
- `original_test_good_medium_overlap`: n=7492, acc=0.8093, macro-F1=0.5445, recall good/medium/bad=0.7507/0.8635/0.0000
- `original_all_bad_core_near_boundary`: n=4084, acc=0.9998, macro-F1=0.3333, recall good/medium/bad=0.0000/0.0000/0.9998
- `original_all_bad_outlier_stress`: n=1201, acc=0.7161, macro-F1=0.2782, recall good/medium/bad=0.0000/0.0000/0.7161

## Counts

- Original all 10s+: `32956` windows.
- Original test 10s+: `8477` windows.
- Bad outlier stress is reported separately because dropping it removes most original-test bad windows.

![bucketed confusions](nl_n7188_gm_trim_bad_boundaryblocks_badoutlier_visqrsnarr_a364001dc6cf__calibrated_original_bucketed_confusions.png)
