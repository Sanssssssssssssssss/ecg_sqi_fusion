# Original Bucketed Checkpoint Report

Report-only evaluation. It is not used for Clean/SemiClean/node selection.

## Checkpoint

- Variant: `nl_n11200_gm_trim_bad_boundaryblocks_n10000shell_thinprob_84079be0011f`
- Prediction mode: `rawbad_feature_pc1_qrsprom_widegood_plus_precision_veto`

## Buckets

- `original_all_10s+`: n=32956, acc=0.8248, macro-F1=0.8490, recall good/medium/bad=0.7100/0.9475/0.9483
- `original_test_all_10s+`: n=8477, acc=0.8514, macro-F1=0.7432, recall good/medium/bad=0.8082/0.9202/0.4915
- `original_test_good_medium_only`: n=8066, acc=0.8697, macro-F1=0.5868, recall good/medium/bad=0.8082/0.9202/0.0000
- `original_test_bad_core_near_boundary`: n=119, acc=1.0000, macro-F1=0.3333, recall good/medium/bad=0.0000/0.0000/1.0000
- `original_test_bad_outlier_stress`: n=292, acc=0.2842, macro-F1=0.1476, recall good/medium/bad=0.0000/0.0000/0.2842
- `original_test_drop_bad_outlier_reference`: n=8185, acc=0.8716, macro-F1=0.7631, recall good/medium/bad=0.8082/0.9202/1.0000
- `original_test_good_medium_overlap`: n=7492, acc=0.8597, macro-F1=0.5813, recall good/medium/bad=0.8062/0.9093/0.0000
- `original_all_bad_core_near_boundary`: n=4084, acc=1.0000, macro-F1=0.3333, recall good/medium/bad=0.0000/0.0000/1.0000
- `original_all_bad_outlier_stress`: n=1201, acc=0.7727, macro-F1=0.2906, recall good/medium/bad=0.0000/0.0000/0.7727

## Counts

- Original all 10s+: `32956` windows.
- Original test 10s+: `8477` windows.
- Bad outlier stress is reported separately because dropping it removes most original-test bad windows.

![bucketed confusions](nl_n11200_gm_trim_bad_boundaryblocks__f66de47f51__rawbad_feature_pc1_qrsprom_widego_d198e034_original_bucketed_confusions.png)
