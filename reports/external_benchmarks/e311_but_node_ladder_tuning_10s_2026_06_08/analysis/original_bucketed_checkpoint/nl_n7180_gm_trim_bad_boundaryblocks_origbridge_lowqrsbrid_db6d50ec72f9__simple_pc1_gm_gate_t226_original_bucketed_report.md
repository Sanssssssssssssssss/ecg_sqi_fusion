# Original Bucketed Checkpoint Report

Report-only evaluation. It is not used for Clean/SemiClean/node selection.

## Checkpoint

- Variant: `nl_n7180_gm_trim_bad_boundaryblocks_origbridge_lowqrsbrid_db6d50ec72f9`
- Prediction mode: `simple_pc1_gm_gate_t226`

## Buckets

- `original_all_10s+`: n=32956, acc=0.8317, macro-F1=0.8517, recall good/medium/bad=0.8149/0.8197/0.9099
- `original_test_all_10s+`: n=8477, acc=0.7500, macro-F1=0.5177, recall good/medium/bad=0.9310/0.6701/0.0073
- `original_test_good_medium_only`: n=8066, acc=0.7879, macro-F1=0.5264, recall good/medium/bad=0.9310/0.6701/0.0000
- `original_test_bad_core_near_boundary`: n=119, acc=0.0000, macro-F1=0.0000, recall good/medium/bad=0.0000/0.0000/0.0000
- `original_test_bad_outlier_stress`: n=292, acc=0.0103, macro-F1=0.0068, recall good/medium/bad=0.0000/0.0000/0.0103
- `original_test_drop_bad_outlier_reference`: n=8185, acc=0.7764, macro-F1=0.5224, recall good/medium/bad=0.9310/0.6701/0.0000
- `original_test_good_medium_overlap`: n=7492, acc=0.7716, macro-F1=0.5136, recall good/medium/bad=0.9303/0.6247/0.0000
- `original_all_bad_core_near_boundary`: n=4084, acc=0.9709, macro-F1=0.3284, recall good/medium/bad=0.0000/0.0000/0.9709
- `original_all_bad_outlier_stress`: n=1201, acc=0.7027, macro-F1=0.2751, recall good/medium/bad=0.0000/0.0000/0.7027

## Counts

- Original all 10s+: `32956` windows.
- Original test 10s+: `8477` windows.
- Bad outlier stress is reported separately because dropping it removes most original-test bad windows.

![bucketed confusions](nl_n7180_gm_trim_bad_boundaryblocks_origbridge_lowqrsbrid_db6d50ec72f9__simple_pc1_gm_gate_t226_original_bucketed_confusions.png)
