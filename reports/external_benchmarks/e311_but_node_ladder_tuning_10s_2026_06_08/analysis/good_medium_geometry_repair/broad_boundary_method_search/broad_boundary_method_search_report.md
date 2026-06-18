# Broad Good/Medium Boundary Method Search

Scope: fast diagnostic search on Clean/SemiClean node rows only. Original BUT is report-only and not used here.

Goal: find a small, reusable ECG/SQI rule family that separates large good/medium boundary blocks, rather than adding a new patch per frontier level.

## Best By Validation Selection

| target | method | rule | val_acc | val_good_recall | val_medium_recall | test_acc | test_good_recall | test_medium_recall | all_acc | all_good_recall | all_medium_recall | all_bad_recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N7179_goodmed | logistic_core_sqi | override_model_disagree_only | 1.0000 | 1.0000 | 1.0000 | 0.9918 | 1.0000 | 1.0000 | 0.9979 | 1.0000 | 0.9993 | 0.9919 |
| N7179_goodmed | logistic_core_sqi | override_nonbad | 1.0000 | 1.0000 | 1.0000 | 0.9918 | 1.0000 | 1.0000 | 0.9979 | 1.0000 | 0.9993 | 0.9919 |
| N7179_mediumsafe | logistic_core_sqi | override_nonbad | 0.9975 | 1.0000 | 1.0000 | 0.9703 | 1.0000 | 1.0000 | 0.9935 | 1.0000 | 1.0000 | 0.9706 |
| N7179_mediumsafe | logistic_core_sqi | override_model_disagree_only | 0.9975 | 1.0000 | 1.0000 | 0.9703 | 1.0000 | 1.0000 | 0.9935 | 1.0000 | 1.0000 | 0.9706 |
| N7178_goodmed | logistic_core_sqi | override_nonbad | 0.9975 | 1.0000 | 1.0000 | 0.9703 | 1.0000 | 1.0000 | 0.9935 | 1.0000 | 1.0000 | 0.9706 |
| N7178_goodmed | logistic_core_sqi | override_model_disagree_only | 0.9975 | 1.0000 | 1.0000 | 0.9703 | 1.0000 | 1.0000 | 0.9935 | 1.0000 | 1.0000 | 0.9706 |
| N7179_goodmed | rf_depth4_upper_bound | override_model_disagree_only | 0.9975 | 0.9968 | 1.0000 | 0.9913 | 1.0000 | 0.9992 | 0.9972 | 0.9994 | 0.9979 | 0.9919 |
| N7179_goodmed | rf_depth4_upper_bound | override_nonbad | 0.9975 | 0.9968 | 1.0000 | 0.9913 | 1.0000 | 0.9992 | 0.9972 | 0.9994 | 0.9979 | 0.9919 |
| N7179_mediumsafe | rf_depth4_upper_bound | override_nonbad | 0.9950 | 0.9968 | 1.0000 | 0.9698 | 1.0000 | 0.9992 | 0.9927 | 0.9994 | 0.9986 | 0.9706 |
| N7179_mediumsafe | rf_depth4_upper_bound | override_model_disagree_only | 0.9950 | 0.9968 | 1.0000 | 0.9698 | 1.0000 | 0.9992 | 0.9927 | 0.9994 | 0.9986 | 0.9706 |
| N7179_goodmed | tree_depth3 | override_model_disagree_only | 0.9950 | 0.9968 | 0.9878 | 0.9890 | 1.0000 | 0.9956 | 0.9962 | 0.9994 | 0.9953 | 0.9919 |
| N7179_goodmed | tree_depth3 | override_nonbad | 0.9950 | 0.9968 | 0.9878 | 0.9890 | 1.0000 | 0.9956 | 0.9962 | 0.9994 | 0.9953 | 0.9919 |
| N7179_goodmed | tree_depth2 | override_model_disagree_only | 0.9950 | 0.9968 | 0.9878 | 0.9890 | 1.0000 | 0.9956 | 0.9962 | 0.9994 | 0.9953 | 0.9919 |
| N7179_goodmed | tree_depth2 | override_nonbad | 0.9950 | 0.9968 | 0.9878 | 0.9890 | 1.0000 | 0.9956 | 0.9962 | 0.9994 | 0.9953 | 0.9919 |
| N7179_goodmed | single_pc1_threshold | if non-bad and pc1 <= -2.2600: good else medium | 0.9950 | 0.9968 | 0.9878 | 0.9890 | 1.0000 | 0.9956 | 0.9961 | 0.9993 | 0.9953 | 0.9919 |
| N7179_mediumsafe | tree_depth3 | override_model_disagree_only | 0.9924 | 0.9968 | 0.9878 | 0.9676 | 1.0000 | 0.9956 | 0.9917 | 0.9994 | 0.9960 | 0.9706 |
| N7178_goodmed | tree_depth3 | override_model_disagree_only | 0.9924 | 0.9968 | 0.9878 | 0.9676 | 1.0000 | 0.9956 | 0.9917 | 0.9994 | 0.9960 | 0.9706 |
| N7178_goodmed | single_pc1_threshold | if non-bad and pc1 <= -2.2600: good else medium | 0.9924 | 0.9968 | 0.9878 | 0.9676 | 1.0000 | 0.9956 | 0.9916 | 0.9993 | 0.9960 | 0.9706 |
| N7179_mediumsafe | tree_depth2 | override_nonbad | 0.9924 | 0.9968 | 0.9878 | 0.9676 | 1.0000 | 0.9956 | 0.9917 | 0.9994 | 0.9960 | 0.9706 |
| N7179_mediumsafe | tree_depth2 | override_model_disagree_only | 0.9924 | 0.9968 | 0.9878 | 0.9676 | 1.0000 | 0.9956 | 0.9917 | 0.9994 | 0.9960 | 0.9706 |

## Best By Held-Out Test Accuracy

| target | method | rule | test_acc | test_good_recall | test_medium_recall | all_acc | all_good_recall | all_medium_recall | all_bad_recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N7179_goodmed | logistic_core_sqi | override_model_disagree_only | 0.9918 | 1.0000 | 1.0000 | 0.9979 | 1.0000 | 0.9993 | 0.9919 |
| N7179_goodmed | logistic_core_sqi | override_nonbad | 0.9918 | 1.0000 | 1.0000 | 0.9979 | 1.0000 | 0.9993 | 0.9919 |
| N7179_goodmed | single_pc1_threshold | if non-bad and pc1 <= -2.5403: good else medium | 0.9913 | 1.0000 | 0.9992 | 0.9950 | 0.9941 | 0.9976 | 0.9919 |
| N7179_goodmed | single_pc1_threshold | if non-bad and pc1 <= -2.9476: good else medium | 0.9913 | 0.9986 | 1.0000 | 0.9777 | 0.9486 | 0.9987 | 0.9919 |
| N7179_goodmed | single_pc1_threshold | if non-bad and pc1 <= -2.8507: good else medium | 0.9913 | 0.9993 | 0.9996 | 0.9832 | 0.9634 | 0.9982 | 0.9919 |
| N7179_goodmed | single_pc1_threshold | if non-bad and pc1 <= -2.5400: good else medium | 0.9913 | 1.0000 | 0.9992 | 0.9950 | 0.9941 | 0.9976 | 0.9919 |
| N7179_goodmed | rf_depth4_upper_bound | override_nonbad | 0.9913 | 1.0000 | 0.9992 | 0.9972 | 0.9994 | 0.9979 | 0.9919 |
| N7179_goodmed | rf_depth4_upper_bound | override_model_disagree_only | 0.9913 | 1.0000 | 0.9992 | 0.9972 | 0.9994 | 0.9979 | 0.9919 |
| N7179_goodmed | single_pc1_threshold | if non-bad and pc1 <= -2.7298: good else medium | 0.9910 | 0.9993 | 0.9992 | 0.9893 | 0.9791 | 0.9980 | 0.9919 |
| N7179_goodmed | single_pc1_threshold | if non-bad and pc1 <= -3.0066: good else medium | 0.9910 | 0.9979 | 1.0000 | 0.9720 | 0.9337 | 0.9990 | 0.9919 |
| N7179_goodmed | single_pc1_threshold | if non-bad and pc1 <= -3.1787: good else medium | 0.9908 | 0.9971 | 1.0000 | 0.9547 | 0.8891 | 0.9992 | 0.9919 |
| N7179_goodmed | single_pc1_threshold | if non-bad and pc1 <= -3.0710: good else medium | 0.9908 | 0.9971 | 1.0000 | 0.9662 | 0.9187 | 0.9990 | 0.9919 |
| N7179_goodmed | single_pc1_threshold | if non-bad and pc1 <= -3.1289: good else medium | 0.9908 | 0.9971 | 1.0000 | 0.9605 | 0.9040 | 0.9992 | 0.9919 |
| N7179_goodmed | single_pc1_threshold | if non-bad and pc1 <= -3.2392: good else medium | 0.9903 | 0.9957 | 1.0000 | 0.9488 | 0.8739 | 0.9992 | 0.9919 |
| N7179_goodmed | single_pc1_threshold | if non-bad and pc1 <= -3.2909: good else medium | 0.9893 | 0.9929 | 1.0000 | 0.9430 | 0.8590 | 0.9992 | 0.9919 |
| N7179_goodmed | single_pc1_threshold | if non-bad and pc1 <= -2.2600: good else medium | 0.9890 | 1.0000 | 0.9956 | 0.9961 | 0.9993 | 0.9953 | 0.9919 |
| N7179_goodmed | tree_depth3 | override_nonbad | 0.9890 | 1.0000 | 0.9956 | 0.9962 | 0.9994 | 0.9953 | 0.9919 |
| N7179_goodmed | single_pc1_threshold | if non-bad and pc1 <= -3.3405: good else medium | 0.9890 | 0.9921 | 1.0000 | 0.9373 | 0.8444 | 0.9992 | 0.9919 |
| N7179_goodmed | tree_depth3 | override_model_disagree_only | 0.9890 | 1.0000 | 0.9956 | 0.9962 | 0.9994 | 0.9953 | 0.9919 |
| N7179_goodmed | tree_depth2 | override_model_disagree_only | 0.9890 | 1.0000 | 0.9956 | 0.9962 | 0.9994 | 0.9953 | 0.9919 |

## Working Interpretation

- The large-block good/medium separation is not an SNR-only problem. The dominant signal is a PCA geometry axis (`pc1`) built from target-aware SQI/morphology features.
- The preferred simple method is now the one-dimensional rule family: keep the model's bad guard, then classify non-bad rows as good when `pc1` is below a frozen threshold and medium otherwise.
- Logistic and random-forest variants are kept as upper-bound diagnostics. They are not the main proposal because the shallow tree and explicit threshold search show that almost all of the recoverable gain is available from one interpretable axis.
- Next validation should freeze the PCA transform and threshold from train/reference rows only, then evaluate held-out Clean/SemiClean and bucketed original report-only. If original still fails, the bottleneck is domain shift, not lack of boundary-block data.