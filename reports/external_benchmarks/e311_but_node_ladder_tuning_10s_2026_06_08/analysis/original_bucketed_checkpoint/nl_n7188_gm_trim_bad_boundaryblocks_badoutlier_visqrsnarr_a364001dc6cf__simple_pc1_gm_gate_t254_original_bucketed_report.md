# Original Bucketed Checkpoint Report

Report-only evaluation. It is not used for Clean/SemiClean/node selection.

## Checkpoint

- Variant: `nl_n7188_gm_trim_bad_boundaryblocks_badoutlier_visqrsnarr_a364001dc6cf`
- Prediction mode: `simple_pc1_gm_gate_t254`

## Buckets

- `original_all_10s+`: n=32956, acc=0.8188, macro-F1=0.8416, recall good/medium/bad=0.7733/0.8340/0.9353
- `original_test_all_10s+`: n=8477, acc=0.7689, macro-F1=0.6479, recall good/medium/bad=0.9173/0.6884/0.3212
- `original_test_good_medium_only`: n=8066, acc=0.7917, macro-F1=0.5331, recall good/medium/bad=0.9173/0.6884/0.0000
- `original_test_bad_core_near_boundary`: n=119, acc=1.0000, macro-F1=0.3333, recall good/medium/bad=0.0000/0.0000/1.0000
- `original_test_bad_outlier_stress`: n=292, acc=0.0445, macro-F1=0.0284, recall good/medium/bad=0.0000/0.0000/0.0445
- `original_test_drop_bad_outlier_reference`: n=8185, acc=0.7947, macro-F1=0.7289, recall good/medium/bad=0.9173/0.6884/1.0000
- `original_test_good_medium_overlap`: n=7492, acc=0.7776, macro-F1=0.5219, recall good/medium/bad=0.9164/0.6491/0.0000
- `original_all_bad_core_near_boundary`: n=4084, acc=0.9998, macro-F1=0.3333, recall good/medium/bad=0.0000/0.0000/0.9998
- `original_all_bad_outlier_stress`: n=1201, acc=0.7161, macro-F1=0.2782, recall good/medium/bad=0.0000/0.0000/0.7161

## Counts

- Original all 10s+: `32956` windows.
- Original test 10s+: `8477` windows.
- Bad outlier stress is reported separately because dropping it removes most original-test bad windows.

![bucketed confusions](nl_n7188_gm_trim_bad_boundaryblocks_badoutlier_visqrsnarr_a364001dc6cf__simple_pc1_gm_gate_t254_original_bucketed_confusions.png)
