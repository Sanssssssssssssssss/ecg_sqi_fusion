# BUT 10s Medium-Guard Bad-Boundary Analysis

Formal result remains original BUT 10s P1 test. Balanced and stratified-balanced subsets are prediction-independent diagnostics.

| rank | anchor | mode | acc | macro-F1 | recalls G/M/B | stratified macro |
| --- | --- | --- | ---: | ---: | --- | ---: |
| 1 | `h_bad_rescue_05` | hypothesis | 0.8229 | 0.7454 | 0.887/0.773/0.793 | 0.8310 |
| 2 | `paper_table_strict__em_ma_bw_badstrong__cinc0p00__cw1p00_1p40_1p70` | full | 0.7940 | 0.7177 | 0.857/0.762/0.577 | 0.7357 |
| 3 | `paper_table_strict__em_ma_ma_heavy__cinc0p00__cw1p00_1p40_1p70` | full | 0.8137 | 0.6956 | 0.861/0.799/0.550 | 0.7462 |
| 4 | `paper_table_strict__em_ma_ma_heavy__cinc0p00__cw1p00_1p55_1p70` | full | 0.7667 | 0.6794 | 0.910/0.657/0.676 | 0.7344 |
| 5 | `paper_table_strict__em_ma_bw_badstrong__cinc0p00__cw1p00_1p55_1p70` | full | 0.7157 | 0.6516 | 0.580/0.845/0.523 | 0.6616 |
| 6 | `paper_denoise_interval__em_ma_ma_heavy__cinc0p00__cw1p00_1p40_1p70` | quick | 0.5918 | 0.5235 | 0.781/0.418/0.791 | 0.6494 |
| 7 | `paper_denoise_interval__em_ma_ma_heavy__cinc0p00__cw1p00_1p55_1p70` | quick | 0.5798 | 0.5155 | 0.791/0.384/0.820 | 0.6531 |

## Next-rule interpretation

- If `bad_missed` is high with visible QRS, add `visible_qrs_but_unusable` and baseline/platform subtypes rather than only lower SNR.
- If `medium_to_bad` is high, reduce medium contact/flat and keep medium QRS preserve high.
- If SQI PCA separates bad better than morphology PCA, promote SQI fusion as an analysis branch only after checking medium recall.
