# Paper Table Trend Comparison

Artifacts: `E:\GPTProject2\ecg\outputs\sqi_paper_aligned`

Reference values are Clifford et al. 2012 Tables 5, 6, and 7.
Because this repo currently uses Set-a-only internal splits, absolute Ac/Se/Sp values are not expected to match exactly; rank/trend is the primary diagnostic.

## Diagnostics

- Table 5 rank Spearman=0.786; top3 overlap=2/3.
- Table 6 rank Spearman=0.600; run best=Quintuplets; paper best=Quintuplets.
- Table 7 run best=SVM; paper balanced Set-b best=MLP by 0.006 Ac.

## Table 5 Single SQI

| SQI | paper_Ac_test | run_Ac_test | delta_Ac_test | paper_rank | run_rank |
| --- | --- | --- | --- | --- | --- |
| bSQI | 0.909 | 0.874 | -0.035 | 3 | 4 |
| iSQI | 0.805 | 0.810 | 0.005 | 5 | 5 |
| kSQI | 0.925 | 0.913 | -0.012 | 2 | 1 |
| sSQI | 0.906 | 0.896 | -0.010 | 4 | 2 |
| pSQI | 0.723 | 0.658 | -0.065 | 7 | 6 |
| fSQI | 0.730 | 0.580 | -0.150 | 6 | 7 |
| basSQI | 0.933 | 0.883 | -0.050 | 1 | 3 |

- Main match: `kSQI`, `sSQI`, and `basSQI` remain among the strongest individual SQIs; `iSQI`, `pSQI`, and `fSQI` remain weaker as single features.
- Main mismatch: `basSQI` is lower than paper by about 0.050 Ac, and `fSQI` is much lower by about 0.150 Ac in the fixed-parameter SVM table.

## Table 6 SQI Combinations

| Group | paper_Ac_test | run_Ac_test | delta_Ac_test | paper_rank | run_rank |
| --- | --- | --- | --- | --- | --- |
| Pairs | 0.945 | 0.900 | -0.045 | 6 | 6 |
| Triplets | 0.948 | 0.931 | -0.017 | 2 | 5 |
| Quadruplets | 0.948 | 0.944 | -0.004 | 3 | 2 |
| Quintuplets | 0.949 | 0.948 | -0.001 | 1 | 1 |
| Sextuplets | 0.948 | 0.935 | -0.013 | 4 | 4 |
| All SQI | 0.946 | 0.939 | -0.007 | 5 | 3 |

- Strong match: `Quintuplets` is best in both paper and this run, and `All SQI` is slightly worse than the selected five-SQI set.
- This is the most important trend check for the paper-aligned line; it suggests the QRS/SQI/model pipeline is now directionally sane.

## Table 7 Selected Five

| Model | paper_Ac_test_balanced_SetB | run_Ac_test | delta_Ac_test | run_Se_test | run_Sp_test |
| --- | --- | --- | --- | --- | --- |
| MLP | 0.959 | 0.918 | -0.041 | 0.931 | 0.904 |
| SVM | 0.953 | 0.939 | -0.014 | 0.966 | 0.913 |

- SVM is close to the paper balanced Set-b number, despite this run using only an internal Set-a group split.
- MLP is lower and reverses the paper's small MLP-over-SVM ordering. The likely cause is protocol instability from the local LM-MLP implementation and proxy split, not a QRS detector failure, because the SVM Table 6 trend is aligned.

## Current Diagnosis

- No blocking implementation bug remains in QRS execution: both official detectors run, all 1546 records have cached annotations, and the cache summary has 1546 x 12 rows.
- The earlier `eplimited` issue was real: without an 8 s warmup, the official EP Limited detector only emitted beats near the end of 10 s records. The adapter now prepends warmup and maps annotation times back to the original segment.
- Remaining trend gaps are most plausibly methodological: Set-a-only labels/splits, synthetic poor cases generated from acceptable Set-a records, no Set-b external test, and not fully identical PSD/noise definitions.
- The specific feature to audit next is `fSQI`: in the fixed SVM single-SQI table it has very high sensitivity but poor specificity and low AUC, while the paper also treats it as weak alone but not this weak.
- fSQI distribution check: median mean-fSQI is 0.1252 for clean unacceptable, 0.0050 for clean acceptable, and 0.0013 for synthetic noisy poor. This explains why fSQI is unstable alone: the two poor-case mechanisms point in different flatline directions.

## Expected Paper Trends

- Table 5 individual SQIs: strongest test Ac are `basSQI`, `kSQI`, then `bSQI/sSQI`; `pSQI` and `fSQI` are weak alone but complementary.
- Table 6 combinations: selected combinations are tightly clustered; `Quintuplets` is best at 0.949 test Ac, while `All SQI` is slightly lower.
- Table 7 selected five SQIs: balanced 12-lead MLP and SVM should be around mid-0.95 Ac on Set-b in the original Set-a/Set-b protocol.
