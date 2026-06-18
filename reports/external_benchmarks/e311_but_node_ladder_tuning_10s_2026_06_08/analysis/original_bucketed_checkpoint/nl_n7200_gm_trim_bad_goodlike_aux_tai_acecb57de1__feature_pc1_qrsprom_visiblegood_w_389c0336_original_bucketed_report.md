# Original Bucketed Checkpoint Report

Report-only evaluation. It is not used for Clean/SemiClean/node selection.

## Checkpoint

- Variant: `nl_n7200_gm_trim_bad_goodlike_aux_tail_a12_good128_mid168_837d9498a6ae`
- Prediction mode: `feature_pc1_qrsprom_visiblegood_wavegood_axis2_diagnostic`

## Buckets

- `original_all_10s+`: n=32956, acc=0.8246, macro-F1=0.8525, recall good/medium/bad=0.7272/0.9113/0.9646
- `original_test_all_10s+`: n=8477, acc=0.9102, macro-F1=0.8474, recall good/medium/bad=0.9082/0.9340/0.6715
- `original_test_good_medium_only`: n=8066, acc=0.9224, macro-F1=0.6186, recall good/medium/bad=0.9082/0.9340/0.0000
- `original_test_bad_core_near_boundary`: n=119, acc=1.0000, macro-F1=0.3333, recall good/medium/bad=0.0000/0.0000/1.0000
- `original_test_bad_outlier_stress`: n=292, acc=0.5377, macro-F1=0.2331, recall good/medium/bad=0.0000/0.0000/0.5377
- `original_test_drop_bad_outlier_reference`: n=8185, acc=0.9235, macro-F1=0.8526, recall good/medium/bad=0.9082/0.9340/1.0000
- `original_test_good_medium_overlap`: n=7492, acc=0.9164, macro-F1=0.6152, recall good/medium/bad=0.9073/0.9249/0.0000
- `original_all_bad_core_near_boundary`: n=4084, acc=1.0000, macro-F1=0.3333, recall good/medium/bad=0.0000/0.0000/1.0000
- `original_all_bad_outlier_stress`: n=1201, acc=0.8443, macro-F1=0.3052, recall good/medium/bad=0.0000/0.0000/0.8443

## Counts

- Original all 10s+: `32956` windows.
- Original test 10s+: `8477` windows.
- Bad outlier stress is reported separately because dropping it removes most original-test bad windows.

![bucketed confusions](nl_n7200_gm_trim_bad_goodlike_aux_tai_acecb57de1__feature_pc1_qrsprom_visiblegood_w_389c0336_original_bucketed_confusions.png)
