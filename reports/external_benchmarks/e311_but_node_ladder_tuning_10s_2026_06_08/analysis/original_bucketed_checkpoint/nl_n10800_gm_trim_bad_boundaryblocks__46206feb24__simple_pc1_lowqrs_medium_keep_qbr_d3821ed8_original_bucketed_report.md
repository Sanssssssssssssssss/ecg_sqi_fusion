# Original Bucketed Checkpoint Report

Report-only evaluation. It is not used for Clean/SemiClean/node selection.

## Checkpoint

- Variant: `nl_n10800_gm_trim_bad_boundaryblocks_bigjump_badprobe_n96_d842a837823b`
- Prediction mode: `simple_pc1_lowqrs_medium_keep_qbr030_flat012`

## Buckets

- `original_all_10s+`: n=32956, acc=0.8423, macro-F1=0.8643, recall good/medium/bad=0.7520/0.9544/0.9084
- `original_test_all_10s+`: n=8477, acc=0.8203, macro-F1=0.5587, recall good/medium/bad=0.7385/0.9638/0.0000
- `original_test_good_medium_only`: n=8066, acc=0.8621, macro-F1=0.5711, recall good/medium/bad=0.7385/0.9638/0.0000
- `original_test_bad_core_near_boundary`: n=119, acc=0.0000, macro-F1=0.0000, recall good/medium/bad=0.0000/0.0000/0.0000
- `original_test_bad_outlier_stress`: n=292, acc=0.0000, macro-F1=0.0000, recall good/medium/bad=0.0000/0.0000/0.0000
- `original_test_drop_bad_outlier_reference`: n=8185, acc=0.8496, macro-F1=0.5675, recall good/medium/bad=0.7385/0.9638/0.0000
- `original_test_good_medium_overlap`: n=7492, acc=0.8516, macro-F1=0.5656, recall good/medium/bad=0.7357/0.9589/0.0000
- `original_all_bad_core_near_boundary`: n=4084, acc=0.9706, macro-F1=0.3284, recall good/medium/bad=0.0000/0.0000/0.9706
- `original_all_bad_outlier_stress`: n=1201, acc=0.6969, macro-F1=0.2738, recall good/medium/bad=0.0000/0.0000/0.6969

## Counts

- Original all 10s+: `32956` windows.
- Original test 10s+: `8477` windows.
- Bad outlier stress is reported separately because dropping it removes most original-test bad windows.

![bucketed confusions](nl_n10800_gm_trim_bad_boundaryblocks__46206feb24__simple_pc1_lowqrs_medium_keep_qbr_d3821ed8_original_bucketed_confusions.png)
