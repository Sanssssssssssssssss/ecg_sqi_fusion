# BUT 10s Bad-Boundary Refinement

This pass refines the first grid's useful `b10_all_bad_wearable` anchor.  The goal is to keep BUT bad recall high while recovering medium/good separation.

| rank | spec | class weight | return | BUT acc | BUT bal | macro-F1 | recalls good/medium/bad | PTB acc | PTB bad | denoise |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 1 | r08_b10_bad_prior_mild | 1,1.42,1.60 | 0 | 0.7654 | 0.7943 | 0.7071 | 0.926/0.627/0.830 | 0.9664 | 1.0000 | 2.560 |
| 2 | r02_b10_cw180 | 1,1.45,1.80 | 0 | 0.7569 | 0.7859 | 0.7040 | 0.800/0.713/0.844 | 0.9764 | 0.9986 | 2.575 |
| 3 | r05_b10_dropout_medium_guard | 1,1.55,1.75 | 0 | 0.7214 | 0.7586 | 0.6846 | 0.995/0.490/0.791 | 0.9800 | 1.0000 | 2.593 |
| 4 | r03_b10_medium_detail | 1,1.55,1.75 | 0 | 0.6764 | 0.7028 | 0.6648 | 0.998/0.409/0.701 | 0.9859 | 0.9986 | 2.652 |
| 5 | r06_b04_soft_medium_guard | 1,1.55,1.70 | 0 | 0.6448 | 0.6952 | 0.6384 | 0.837/0.475/0.774 | 0.9877 | 1.0000 | 2.915 |
| 6 | r07_b10_good_lenient | 1,1.45,1.70 | 0 | 0.5516 | 0.6584 | 0.5359 | 0.487/0.571/0.917 | 0.9818 | 0.9986 | 2.856 |
| 7 | r04_b10_less_bad_bias | 1,1.45,1.75 | 0 | 0.7688 | 0.6541 | 0.6647 | 0.907/0.692/0.363 | 0.9732 | 0.9986 | 2.680 |
| 8 | r01_b10_cw170 | 1,1.45,1.70 | 0 | 0.6762 | 0.6350 | 0.6001 | 0.948/0.470/0.487 | 0.9800 | 0.9986 | 2.606 |

## Reading Guide

- `r01/r02` isolate class-weight pressure on the successful b10 rule.
- `r03/r05/r06` make medium more explicitly 'detail unreliable but QRS detectable', so medium should stop being swallowed by bad.
- `r04/r08` soften bad severity to test whether b10 is over-triggering good/medium.
- `r07` makes good less artificially pristine, testing whether BUT good is being rejected because our synthetic good is too clean.
