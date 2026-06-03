# BUT 10s Bad-Boundary Tuning

Formal protocol is fixed to 10s (`p1_current_10s_center`).  These runs change PTB synthetic bad/medium artifact rules, then train Uformer quick recipes.

## Results Ranked By Bad Recall

| rank | spec | return | BUT acc | BUT bal | BUT macro-F1 | BUT recalls good/medium/bad | PTB acc | PTB bad recall |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 1 | b04_baseline_step_dropout | 0 | 0.6319 | 0.7246 | 0.6127 | 0.988/0.317/0.869 | 0.9696 | 0.9986 |
| 2 | b10_all_bad_wearable | 0 | 0.7735 | 0.8045 | 0.7238 | 0.824/0.724/0.866 | 0.9768 | 0.9986 |
| 3 | b03_flatline_contact | 0 | 0.6178 | 0.6582 | 0.5882 | 0.752/0.498/0.725 | 0.9746 | 1.0000 |
| 4 | b07_but_mixed_bad | 0 | 0.6888 | 0.6740 | 0.6314 | 0.775/0.624/0.623 | 0.9723 | 0.9973 |
| 5 | b06_spurious_qrs_burst | 0 | 0.6321 | 0.5424 | 0.5531 | 0.854/0.481/0.292 | 0.9641 | 0.9959 |
| 6 | b05_clipping_lowamp | 0 | 0.6695 | 0.5440 | 0.5422 | 0.988/0.451/0.192 | 0.9659 | 0.9973 |

## Decision Notes

- A useful direction must beat the 10s balanced baseline bad recall 0.805 without collapsing medium.
- If BUT bad recall improves only by making every flat/low-amplitude segment bad, check good false-bad galleries before promotion.
- If PTB acc falls below 0.975 or PTB bad recall below 0.985, keep the spec as diagnostic only.
