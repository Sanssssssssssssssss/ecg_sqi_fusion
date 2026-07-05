# BUT Time-Local Latent Transplant

- Rescue pairs: `123`
- Self/empty H-patch max logit error: `0`
- Verdict: **supports the high-resolution query-fusion path, but not a uniquely top-local transplant effect**.
- Figure: `outputs\transformer\supplemental\chapter4_evidence_frozen_final\figures\fig_M6_but_time_local_latent_transplant.png`

## Transplant summary

| direction | intervention | n | mean_delta_h_med | sem_delta_h_med | flip_rate | mean_delta_bad_logit |
| --- | --- | --- | --- | --- | --- | --- |
| necessity | top-local | 123 | 0.0303 | 0.0205 | 0.0163 | 0.0052 |
| necessity | random | 123 | 0.0275 | 0.0151 | 0.0000 | 0.0110 |
| necessity | same-class | 123 | -0.0241 | 0.0151 | 0.0000 | -0.0152 |
| necessity | donor global-mean | 123 | -0.0241 | 0.0092 | 0.0081 | -0.0481 |
| sufficiency | top-local | 123 | 0.0034 | 0.0151 | 0.0163 | -0.0086 |
| sufficiency | random | 123 | 0.0151 | 0.0113 | 0.0244 | -0.0035 |
| sufficiency | same-class | 123 | -0.0103 | 0.0232 | 0.0081 | -0.0130 |
| sufficiency | donor global-mean | 123 | 0.0321 | 0.0104 | 0.0081 | 0.0241 |

## Hi-res path ablation

| condition | n_pairs | boundary_error_rate | medium_recall | good_recall | mean_delta_bad_logit |
| --- | --- | --- | --- | --- | --- |
| full | 123 | 0.0976 | 1.0000 | 0.8049 | 0.0000 |
| no hi-res cross attention | 123 | 0.1951 | 0.9919 | 0.6179 | 1.5893 |
| shuffled H | 123 | 0.0976 | 1.0000 | 0.8049 | -0.0000 |
| mean H | 123 | 0.1870 | 0.9919 | 0.6341 | 1.1637 |
