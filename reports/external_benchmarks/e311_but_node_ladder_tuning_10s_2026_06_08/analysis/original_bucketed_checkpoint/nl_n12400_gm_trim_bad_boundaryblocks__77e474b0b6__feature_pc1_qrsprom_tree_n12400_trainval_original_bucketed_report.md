# Original Bucketed Checkpoint Report

Report-only evaluation. It is not used for Clean/SemiClean/node selection.

## Checkpoint

- Variant: `nl_n12400_gm_trim_bad_boundaryblocks_large_badcore_goodpr_0b11e014ad46`
- Prediction mode: `feature_pc1_qrsprom_tree_n12400_trainval`

## Buckets

- `original_all_10s+`: n=32956, acc=0.8598, macro-F1=0.8794, recall good/medium/bad=0.8044/0.9132/0.9313
- `original_test_all_10s+`: n=8477, acc=0.8432, macro-F1=0.7194, recall good/medium/bad=0.8074/0.9241/0.2895
- `original_test_good_medium_only`: n=8066, acc=0.8714, macro-F1=0.5792, recall good/medium/bad=0.8074/0.9241/0.0000
- `original_test_bad_core_near_boundary`: n=119, acc=1.0000, macro-F1=0.3333, recall good/medium/bad=0.0000/0.0000/1.0000
- `original_test_bad_outlier_stress`: n=292, acc=0.0000, macro-F1=0.0000, recall good/medium/bad=0.0000/0.0000/0.0000
- `original_test_drop_bad_outlier_reference`: n=8185, acc=0.8733, macro-F1=0.9125, recall good/medium/bad=0.8074/0.9241/1.0000
- `original_test_good_medium_overlap`: n=7492, acc=0.8616, macro-F1=0.5737, recall good/medium/bad=0.8054/0.9136/0.0000
- `original_all_bad_core_near_boundary`: n=4084, acc=1.0000, macro-F1=0.3333, recall good/medium/bad=0.0000/0.0000/1.0000
- `original_all_bad_outlier_stress`: n=1201, acc=0.6978, macro-F1=0.2740, recall good/medium/bad=0.0000/0.0000/0.6978

## Counts

- Original all 10s+: `32956` windows.
- Original test 10s+: `8477` windows.
- Bad outlier stress is reported separately because dropping it removes most original-test bad windows.

![bucketed confusions](nl_n12400_gm_trim_bad_boundaryblocks__77e474b0b6__feature_pc1_qrsprom_tree_n12400_trainval_original_bucketed_confusions.png)
