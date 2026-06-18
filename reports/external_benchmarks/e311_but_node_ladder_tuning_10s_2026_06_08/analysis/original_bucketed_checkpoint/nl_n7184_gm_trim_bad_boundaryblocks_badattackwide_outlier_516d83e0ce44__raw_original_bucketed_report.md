# Original Bucketed Checkpoint Report

Report-only evaluation. It is not used for Clean/SemiClean/node selection.

## Checkpoint

- Variant: `nl_n7184_gm_trim_bad_boundaryblocks_badattackwide_outlier_516d83e0ce44`
- Prediction mode: `raw`

## Buckets

- `original_all_10s+`: n=32956, acc=0.8009, macro-F1=0.8100, recall good/medium/bad=0.8673/0.6280/0.9343
- `original_test_all_10s+`: n=8477, acc=0.6647, macro-F1=0.5508, recall good/medium/bad=0.7679/0.6114/0.3260
- `original_test_good_medium_only`: n=8066, acc=0.6820, macro-F1=0.4674, recall good/medium/bad=0.7679/0.6114/0.0000
- `original_test_bad_core_near_boundary`: n=119, acc=0.8403, macro-F1=0.3044, recall good/medium/bad=0.0000/0.0000/0.8403
- `original_test_bad_outlier_stress`: n=292, acc=0.1164, macro-F1=0.0695, recall good/medium/bad=0.0000/0.0000/0.1164
- `original_test_drop_bad_outlier_reference`: n=8185, acc=0.6843, macro-F1=0.5692, recall good/medium/bad=0.7679/0.6114/0.8403
- `original_test_good_medium_overlap`: n=7492, acc=0.6623, macro-F1=0.4533, recall good/medium/bad=0.7654/0.5668/0.0000
- `original_all_bad_core_near_boundary`: n=4084, acc=0.9951, macro-F1=0.3325, recall good/medium/bad=0.0000/0.0000/0.9951
- `original_all_bad_outlier_stress`: n=1201, acc=0.7277, macro-F1=0.2808, recall good/medium/bad=0.0000/0.0000/0.7277

## Counts

- Original all 10s+: `32956` windows.
- Original test 10s+: `8477` windows.
- Bad outlier stress is reported separately because dropping it removes most original-test bad windows.

![bucketed confusions](nl_n7184_gm_trim_bad_boundaryblocks_badattackwide_outlier_516d83e0ce44__raw_original_bucketed_confusions.png)
