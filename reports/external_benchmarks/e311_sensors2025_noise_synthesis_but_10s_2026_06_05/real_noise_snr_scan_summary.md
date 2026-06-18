# Sensors 2025-Style Noise Synthesis Scan vs BUT 10s P1

This scan reproduces the paper's data idea only: real NSTDB EM/MA noise determines target SNR; BW may be overlaid but is excluded from target SNR.
Our model input remains 10s@125Hz. The paper's 5s@200Hz model/protocol is documented but not used here.

| rank | variant | SNR profile | mix | CinC bad frac | score | morph | medium | SQI | domain bal |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `paper_denoise_interval__em_ma_ma_heavy__cinc0p00__cw1p00_1p40_1p70` | paper_denoise_interval | em_ma_ma_heavy | 0.00 | 0.4733 | 0.4533 | 0.3284 | 0.4107 | 0.9127 |
| 2 | `paper_denoise_interval__em_ma_ma_heavy__cinc0p00__cw1p00_1p55_1p70` | paper_denoise_interval | em_ma_ma_heavy | 0.00 | 0.4733 | 0.4533 | 0.3284 | 0.4107 | 0.9127 |
| 3 | `paper_denoise_interval__em_ma_bw_light__cinc0p00__cw1p00_1p40_1p70` | paper_denoise_interval | em_ma_bw_light | 0.00 | 0.4774 | 0.4584 | 0.3288 | 0.4306 | 0.8968 |
| 4 | `paper_denoise_interval__em_ma_bw_light__cinc0p00__cw1p00_1p55_1p70` | paper_denoise_interval | em_ma_bw_light | 0.00 | 0.4774 | 0.4584 | 0.3288 | 0.4306 | 0.8968 |
| 5 | `paper_table_strict__em_ma_bw_light__cinc0p00__cw1p00_1p40_1p70` | paper_table_strict | em_ma_bw_light | 0.00 | 0.4778 | 0.4662 | 0.3345 | 0.4087 | 0.8968 |
| 6 | `paper_table_strict__em_ma_bw_light__cinc0p00__cw1p00_1p55_1p70` | paper_table_strict | em_ma_bw_light | 0.00 | 0.4778 | 0.4662 | 0.3345 | 0.4087 | 0.8968 |
| 7 | `paper_table_strict__em_ma_bw_badstrong__cinc0p00__cw1p00_1p40_1p70` | paper_table_strict | em_ma_bw_badstrong | 0.00 | 0.4779 | 0.4632 | 0.3291 | 0.4226 | 0.8968 |
| 8 | `paper_table_strict__em_ma_bw_badstrong__cinc0p00__cw1p00_1p55_1p70` | paper_table_strict | em_ma_bw_badstrong | 0.00 | 0.4779 | 0.4632 | 0.3291 | 0.4226 | 0.8968 |
| 9 | `paper_denoise_interval__em_ma_bw_mid__cinc0p00__cw1p00_1p40_1p70` | paper_denoise_interval | em_ma_bw_mid | 0.00 | 0.4780 | 0.4586 | 0.3293 | 0.4325 | 0.8968 |
| 10 | `paper_denoise_interval__em_ma_bw_mid__cinc0p00__cw1p00_1p55_1p70` | paper_denoise_interval | em_ma_bw_mid | 0.00 | 0.4780 | 0.4586 | 0.3293 | 0.4325 | 0.8968 |
| 11 | `paper_table_strict__em_ma_ma_heavy__cinc0p00__cw1p00_1p40_1p70` | paper_table_strict | em_ma_ma_heavy | 0.00 | 0.4782 | 0.4617 | 0.3346 | 0.4048 | 0.9206 |
| 12 | `paper_table_strict__em_ma_ma_heavy__cinc0p00__cw1p00_1p55_1p70` | paper_table_strict | em_ma_ma_heavy | 0.00 | 0.4782 | 0.4617 | 0.3346 | 0.4048 | 0.9206 |
| 13 | `paper_table_strict__em_ma_bw_mid__cinc0p00__cw1p00_1p40_1p70` | paper_table_strict | em_ma_bw_mid | 0.00 | 0.4791 | 0.4669 | 0.3368 | 0.4107 | 0.8968 |
| 14 | `paper_table_strict__em_ma_bw_mid__cinc0p00__cw1p00_1p55_1p70` | paper_table_strict | em_ma_bw_mid | 0.00 | 0.4791 | 0.4669 | 0.3368 | 0.4107 | 0.8968 |
| 15 | `paper_denoise_interval__ma_only__cinc0p00__cw1p00_1p40_1p70` | paper_denoise_interval | ma_only | 0.00 | 0.4791 | 0.4522 | 0.3367 | 0.4266 | 0.9206 |
| 16 | `paper_denoise_interval__ma_only__cinc0p00__cw1p00_1p55_1p70` | paper_denoise_interval | ma_only | 0.00 | 0.4791 | 0.4522 | 0.3367 | 0.4266 | 0.9206 |
| 17 | `paper_denoise_interval__em_ma_bw_mid__cinc0p10__cw1p00_1p40_1p70` | paper_denoise_interval | em_ma_bw_mid | 0.10 | 0.4803 | 0.4619 | 0.3286 | 0.4325 | 0.9048 |
| 18 | `paper_denoise_interval__em_ma_bw_mid__cinc0p10__cw1p00_1p55_1p70` | paper_denoise_interval | em_ma_bw_mid | 0.10 | 0.4803 | 0.4619 | 0.3286 | 0.4325 | 0.9048 |
| 19 | `paper_denoise_interval__em_ma_bw_light__cinc0p10__cw1p00_1p40_1p70` | paper_denoise_interval | em_ma_bw_light | 0.10 | 0.4808 | 0.4619 | 0.3289 | 0.4345 | 0.9048 |
| 20 | `paper_denoise_interval__em_ma_bw_light__cinc0p10__cw1p00_1p55_1p70` | paper_denoise_interval | em_ma_bw_light | 0.10 | 0.4808 | 0.4619 | 0.3289 | 0.4345 | 0.9048 |
| 21 | `paper_table_strict__ma_only__cinc0p00__cw1p00_1p55_1p70` | paper_table_strict | ma_only | 0.00 | 0.4809 | 0.4607 | 0.3409 | 0.4127 | 0.9206 |
| 22 | `paper_denoise_interval__em_ma_equal__cinc0p00__cw1p00_1p40_1p70` | paper_denoise_interval | em_ma_equal | 0.00 | 0.4812 | 0.4586 | 0.3332 | 0.4226 | 0.9286 |
| 23 | `paper_denoise_interval__em_ma_equal__cinc0p00__cw1p00_1p55_1p70` | paper_denoise_interval | em_ma_equal | 0.00 | 0.4812 | 0.4586 | 0.3332 | 0.4226 | 0.9286 |
| 24 | `paper_table_strict__em_ma_bw_light__cinc0p10__cw1p00_1p40_1p70` | paper_table_strict | em_ma_bw_light | 0.10 | 0.4812 | 0.4691 | 0.3378 | 0.4107 | 0.9048 |
| 25 | `paper_table_strict__em_ma_bw_light__cinc0p10__cw1p00_1p55_1p70` | paper_table_strict | em_ma_bw_light | 0.10 | 0.4812 | 0.4691 | 0.3378 | 0.4107 | 0.9048 |
| 26 | `paper_denoise_interval__em_ma_bw_badstrong__cinc0p00__cw1p00_1p40_1p70` | paper_denoise_interval | em_ma_bw_badstrong | 0.00 | 0.4814 | 0.4613 | 0.3324 | 0.4246 | 0.9206 |
| 27 | `paper_denoise_interval__em_ma_bw_badstrong__cinc0p00__cw1p00_1p55_1p70` | paper_denoise_interval | em_ma_bw_badstrong | 0.00 | 0.4814 | 0.4613 | 0.3324 | 0.4246 | 0.9206 |
| 28 | `paper_denoise_interval__em_ma_bw_badstrong__cinc0p10__cw1p00_1p40_1p70` | paper_denoise_interval | em_ma_bw_badstrong | 0.10 | 0.4817 | 0.4633 | 0.3350 | 0.4286 | 0.9048 |
| 29 | `paper_denoise_interval__em_ma_bw_badstrong__cinc0p10__cw1p00_1p55_1p70` | paper_denoise_interval | em_ma_bw_badstrong | 0.10 | 0.4817 | 0.4633 | 0.3350 | 0.4286 | 0.9048 |
| 30 | `paper_table_strict__em_ma_bw_mid__cinc0p25__cw1p00_1p40_1p70` | paper_table_strict | em_ma_bw_mid | 0.25 | 0.4817 | 0.4711 | 0.3342 | 0.4187 | 0.8968 |

## Checks

- `target_snr_em_ma_db` and `measured_snr_all_noise_db` are both recorded in generated labels.
- BW overlay is reported via `bw_overlay_scale`; it is not used in target SNR scaling.
- Lower distance score only selects candidates for training; BUT test thresholds are never selected on test.
