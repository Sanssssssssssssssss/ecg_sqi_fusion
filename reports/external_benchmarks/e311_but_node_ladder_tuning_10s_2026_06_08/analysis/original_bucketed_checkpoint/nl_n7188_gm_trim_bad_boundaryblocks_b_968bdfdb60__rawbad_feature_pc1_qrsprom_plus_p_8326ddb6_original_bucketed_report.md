# Original Bucketed Checkpoint Report

Report-only evaluation. It is not used for Clean/SemiClean/node selection.

## Checkpoint

- Variant: `nl_n7188_gm_trim_bad_boundaryblocks_badoutlier_visqrsnarr_a364001dc6cf`
- Prediction mode: `rawbad_feature_pc1_qrsprom_plus_precision_veto`

## Buckets

- `original_all_10s+`: n=32956, acc=0.8337, macro-F1=0.8549, recall good/medium/bad=0.7344/0.9349/0.9502
- `original_test_all_10s+`: n=8477, acc=0.8364, macro-F1=0.7335, recall good/medium/bad=0.7788/0.9141/0.5085
- `original_test_good_medium_only`: n=8066, acc=0.8531, macro-F1=0.5758, recall good/medium/bad=0.7788/0.9141/0.0000
- `original_test_bad_core_near_boundary`: n=119, acc=1.0000, macro-F1=0.3333, recall good/medium/bad=0.0000/0.0000/1.0000
- `original_test_bad_outlier_stress`: n=292, acc=0.3082, macro-F1=0.1571, recall good/medium/bad=0.0000/0.0000/0.3082
- `original_test_drop_bad_outlier_reference`: n=8185, acc=0.8552, macro-F1=0.7457, recall good/medium/bad=0.7788/0.9141/1.0000
- `original_test_good_medium_overlap`: n=7492, acc=0.8437, macro-F1=0.5704, recall good/medium/bad=0.7765/0.9059/0.0000
- `original_all_bad_core_near_boundary`: n=4084, acc=1.0000, macro-F1=0.3333, recall good/medium/bad=0.0000/0.0000/1.0000
- `original_all_bad_outlier_stress`: n=1201, acc=0.7810, macro-F1=0.2923, recall good/medium/bad=0.0000/0.0000/0.7810

## Counts

- Original all 10s+: `32956` windows.
- Original test 10s+: `8477` windows.
- Bad outlier stress is reported separately because dropping it removes most original-test bad windows.

![bucketed confusions](nl_n7188_gm_trim_bad_boundaryblocks_b_968bdfdb60__rawbad_feature_pc1_qrsprom_plus_p_8326ddb6_original_bucketed_confusions.png)
