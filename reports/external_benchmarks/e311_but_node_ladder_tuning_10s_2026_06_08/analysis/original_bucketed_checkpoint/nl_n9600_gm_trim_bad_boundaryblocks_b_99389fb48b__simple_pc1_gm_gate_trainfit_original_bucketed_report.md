# Original Bucketed Checkpoint Report

Report-only evaluation. It is not used for Clean/SemiClean/node selection.

## Checkpoint

- Variant: `nl_n9600_gm_trim_bad_boundaryblocks_bigjump_balanced_n840_26f327936c0d`
- Prediction mode: `simple_pc1_gm_gate_trainfit`

## Buckets

- `original_all_10s+`: n=32956, acc=0.8194, macro-F1=0.8437, recall good/medium/bad=0.7779/0.8416/0.9090
- `original_test_all_10s+`: n=8477, acc=0.7604, macro-F1=0.5217, recall good/medium/bad=0.9198/0.6997/0.0024
- `original_test_good_medium_only`: n=8066, acc=0.7990, macro-F1=0.5334, recall good/medium/bad=0.9198/0.6997/0.0000
- `original_test_bad_core_near_boundary`: n=119, acc=0.0000, macro-F1=0.0000, recall good/medium/bad=0.0000/0.0000/0.0000
- `original_test_bad_outlier_stress`: n=292, acc=0.0034, macro-F1=0.0023, recall good/medium/bad=0.0000/0.0000/0.0034
- `original_test_drop_bad_outlier_reference`: n=8185, acc=0.7874, macro-F1=0.5294, recall good/medium/bad=0.9198/0.6997/0.0000
- `original_test_good_medium_overlap`: n=7492, acc=0.7836, macro-F1=0.5218, recall good/medium/bad=0.9189/0.6584/0.0000
- `original_all_bad_core_near_boundary`: n=4084, acc=0.9706, macro-F1=0.3284, recall good/medium/bad=0.0000/0.0000/0.9706
- `original_all_bad_outlier_stress`: n=1201, acc=0.6994, macro-F1=0.2744, recall good/medium/bad=0.0000/0.0000/0.6994

## Counts

- Original all 10s+: `32956` windows.
- Original test 10s+: `8477` windows.
- Bad outlier stress is reported separately because dropping it removes most original-test bad windows.

![bucketed confusions](nl_n9600_gm_trim_bad_boundaryblocks_b_99389fb48b__simple_pc1_gm_gate_trainfit_original_bucketed_confusions.png)
