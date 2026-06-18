# Real NSTDB Noise SNR Scan vs BUT 10s P1

This scan uses only real NSTDB `em/ma/bw` noise mixed into PTB clean ECG at class-specific SNR ranges.
No hand-written morphology/contact/pseudo-peak rules are used in these candidates.

- Old PTB-vs-BUT morphology domain separability reference: `0.971`.
- Lower `overall_but_like_score` is better.

## Top Distance Candidates

| rank | variant | mix | SNR G/M/B | score | morph | medium | SQI | domain bal |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `single_bw_snr07_triangular` | single_bw | [16, 20]/[6, 12]/[-6, 0] | 0.4435 | 0.4425 | 0.2719 | 0.4167 | 0.9179 |
| 2 | `single_bw_snr02_uniform` | single_bw | [14, 18]/[6, 12]/[-4, 2] | 0.4453 | 0.4418 | 0.2727 | 0.4345 | 0.9099 |
| 3 | `single_bw_snr03_uniform` | single_bw | [12, 16]/[4, 10]/[-2, 4] | 0.4454 | 0.4401 | 0.2738 | 0.4385 | 0.9111 |
| 4 | `single_bw_snr06_triangular` | single_bw | [12, 16]/[6, 12]/[-2, 4] | 0.4457 | 0.4407 | 0.2719 | 0.4464 | 0.9044 |
| 5 | `single_bw_snr04_uniform` | single_bw | [10, 14]/[2, 8]/[0, 6] | 0.4459 | 0.4409 | 0.2774 | 0.4425 | 0.8976 |
| 6 | `blockwise_switch_snr02_uniform` | blockwise_switch | [14, 18]/[6, 12]/[-4, 2] | 0.4461 | 0.4364 | 0.2705 | 0.4405 | 0.9417 |
| 7 | `blockwise_switch_snr06_triangular` | blockwise_switch | [12, 16]/[6, 12]/[-2, 4] | 0.4461 | 0.4374 | 0.2687 | 0.4444 | 0.9361 |
| 8 | `single_bw_snr08_triangular` | single_bw | [10, 14]/[4, 10]/[-2, 4] | 0.4474 | 0.4445 | 0.2733 | 0.4425 | 0.9044 |
| 9 | `blockwise_switch_snr05_triangular` | blockwise_switch | [14, 18]/[8, 14]/[-4, 2] | 0.4478 | 0.4390 | 0.2729 | 0.4425 | 0.9373 |
| 10 | `single_bw_snr05_triangular` | single_bw | [14, 18]/[8, 14]/[-4, 2] | 0.4482 | 0.4447 | 0.2762 | 0.4385 | 0.9099 |
| 11 | `single_bw_snr01_uniform` | single_bw | [16, 20]/[8, 14]/[-6, 0] | 0.4485 | 0.4451 | 0.2782 | 0.4306 | 0.9179 |
| 12 | `blockwise_switch_snr01_uniform` | blockwise_switch | [16, 20]/[8, 14]/[-6, 0] | 0.4485 | 0.4419 | 0.2712 | 0.4306 | 0.9520 |
| 13 | `dirichlet_balanced_snr01_uniform` | dirichlet_balanced | [16, 20]/[8, 14]/[-6, 0] | 0.4486 | 0.4423 | 0.2788 | 0.4306 | 0.9317 |
| 14 | `bw_heavy_snr01_uniform` | bw_heavy | [16, 20]/[8, 14]/[-6, 0] | 0.4487 | 0.4419 | 0.2758 | 0.4266 | 0.9476 |
| 15 | `blockwise_switch_snr07_triangular` | blockwise_switch | [16, 20]/[6, 12]/[-6, 0] | 0.4488 | 0.4410 | 0.2687 | 0.4306 | 0.9655 |
| 16 | `em_ma_no_bw_snr01_uniform` | em_ma_no_bw | [16, 20]/[8, 14]/[-6, 0] | 0.4493 | 0.4334 | 0.2893 | 0.4266 | 0.9635 |
| 17 | `blockwise_switch_snr03_uniform` | blockwise_switch | [12, 16]/[4, 10]/[-2, 4] | 0.4499 | 0.4398 | 0.2752 | 0.4464 | 0.9429 |
| 18 | `bw_heavy_snr05_triangular` | bw_heavy | [14, 18]/[8, 14]/[-4, 2] | 0.4506 | 0.4397 | 0.2771 | 0.4504 | 0.9397 |
| 19 | `ma_bw_wearable_snr01_uniform` | ma_bw_wearable | [16, 20]/[8, 14]/[-6, 0] | 0.4512 | 0.4372 | 0.2926 | 0.4266 | 0.9544 |
| 20 | `blockwise_switch_snr08_triangular` | blockwise_switch | [10, 14]/[4, 10]/[-2, 4] | 0.4514 | 0.4425 | 0.2715 | 0.4504 | 0.9472 |

## Reading Notes

- If these candidates are closer to BUT but later underperform, pure real-noise SNR is a useful natural-control but not sufficient.
- If medium distance stays high, the BUT medium boundary is not captured by SNR-only noise mixing.
- Test data is not used for calibration or candidate selection; this stage is distribution-only.
