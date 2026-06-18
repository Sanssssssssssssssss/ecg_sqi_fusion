# Sensors 2025-Style Noise Synthesis BUT Model Validation

Rule-based reference `h_bad_rescue_05`: acc `0.8229`, balanced `0.8177`, macro-F1 `0.7454`, recalls `[0.887, 0.773, 0.793]`.

| rank | mode | variant | return | orig macro | orig recalls G/M/B | bal cal macro | bal raw macro | PTB acc | PTB bad | distance |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | full | `paper_table_strict__em_ma_ma_heavy__cinc0p00__cw1p00_1p40_1p70` | 0 | 0.6956 | 0.861/0.799/0.550 | 0.7371 | 0.6990 | 0.8501 | 0.9986 | 0.4782 |
| 2 | quick | `paper_table_strict__em_ma_ma_heavy__cinc0p00__cw1p00_1p55_1p70` | 0 | 0.7065 | 0.886/0.758/0.584 | 0.7326 | 0.7053 | 0.8070 | 0.9687 | 0.4782 |
| 3 | full | `paper_table_strict__em_ma_ma_heavy__cinc0p00__cw1p00_1p55_1p70` | 0 | 0.6794 | 0.910/0.657/0.676 | 0.7291 | 0.6808 | 0.8465 | 0.9986 | 0.4782 |
| 4 | full | `paper_table_strict__em_ma_bw_badstrong__cinc0p00__cw1p00_1p40_1p70` | 0 | 0.7177 | 0.857/0.762/0.577 | 0.7236 | 0.5869 | 0.8928 | 0.9918 | 0.4779 |
| 5 | quick | `paper_table_strict__em_ma_ma_heavy__cinc0p00__cw1p00_1p40_1p70` | 0 | 0.6897 | 0.844/0.769/0.555 | 0.7075 | 0.6734 | 0.8102 | 0.9401 | 0.4782 |
| 6 | quick | `paper_table_strict__em_ma_bw_badstrong__cinc0p00__cw1p00_1p55_1p70` | 0 | 0.6563 | 0.747/0.760/0.637 | 0.7029 | 0.6618 | 0.8692 | 0.9905 | 0.4779 |
| 7 | quick | `paper_table_strict__em_ma_bw_badstrong__cinc0p00__cw1p00_1p40_1p70` | 0 | 0.6969 | 0.764/0.825/0.540 | 0.7015 | 0.6966 | 0.8520 | 0.9905 | 0.4779 |
| 8 | quick | `paper_denoise_interval__em_ma_bw_light__cinc0p00__cw1p00_1p55_1p70` | 0 | 0.6241 | 0.772/0.689/0.625 | 0.6793 | 0.6345 | 0.8206 | 0.9619 | 0.4774 |
| 9 | quick | `paper_table_strict__em_ma_bw_light__cinc0p00__cw1p00_1p40_1p70` | 0 | 0.6380 | 0.710/0.749/0.596 | 0.6714 | 0.6478 | 0.8797 | 0.9850 | 0.4778 |
| 10 | quick | `paper_denoise_interval__em_ma_ma_heavy__cinc0p00__cw1p00_1p55_1p70` | 0 | 0.5155 | 0.791/0.384/0.820 | 0.6389 | 0.6692 | 0.7611 | 0.9196 | 0.4733 |
| 11 | quick | `paper_denoise_interval__em_ma_bw_light__cinc0p00__cw1p00_1p40_1p70` | 0 | 0.5790 | 0.669/0.661/0.693 | 0.6657 | 0.6221 | 0.8115 | 0.9155 | 0.4774 |
| 12 | quick | `paper_table_strict__em_ma_bw_light__cinc0p00__cw1p00_1p55_1p70` | 0 | 0.6479 | 0.646/0.806/0.557 | 0.6638 | 0.6343 | 0.8792 | 0.9918 | 0.4778 |
| 13 | quick | `paper_denoise_interval__em_ma_ma_heavy__cinc0p00__cw1p00_1p40_1p70` | 0 | 0.5235 | 0.781/0.418/0.791 | 0.6550 | 0.6388 | 0.7570 | 0.8951 | 0.4733 |
| 14 | full | `paper_table_strict__em_ma_bw_badstrong__cinc0p00__cw1p00_1p55_1p70` | 0 | 0.6516 | 0.580/0.845/0.523 | 0.6455 | 0.6103 | 0.8837 | 0.9891 | 0.4779 |
| 15 | quick | `paper_denoise_interval__em_ma_bw_mid__cinc0p00__cw1p00_1p55_1p70` | 0 | 0.6136 | 0.835/0.641/0.472 | 0.6355 | 0.6248 | 0.8233 | 0.9319 | 0.4780 |
| 16 | quick | `paper_denoise_interval__em_ma_bw_mid__cinc0p00__cw1p00_1p40_1p70` | 1073807364 | 0.0000 | 0.000/0.000/0.000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.4780 |

## Interpretation Contract

- If paper-style strict SNR loses to `h_bad_rescue_05`, keep it as a natural-noise control rather than forcing it into the mainline.
- If it improves medium while bad stays >=0.80, it supports adding paper-like weak SQI labels to the next generator.
- If BW overlay reduces all-noise SNR but does not improve BUT, that supports the paper's claim that BW should not define SQI by itself.
