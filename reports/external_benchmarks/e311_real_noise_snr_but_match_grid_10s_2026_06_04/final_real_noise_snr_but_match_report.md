# Real NSTDB Noise SNR Model Validation

Rule-based reference `h_bad_rescue_05`: acc `0.8229`, balanced `0.8177`, macro-F1 `0.7454`.

| rank | mode | variant | return | BUT acc | BUT bal | BUT macro | recalls G/M/B | PTB acc | PTB bad | distance score |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 1 | quick | `blockwise_switch_snr05_triangular` | 0 | 0.6761 | 0.6692 | 0.6007 | 0.795/0.583/0.630 | 0.7430 | 0.9973 | 0.4478 |
| 2 | quick | `blockwise_switch_snr02_uniform` | 0 | 0.6133 | 0.6228 | 0.5321 | 0.563/0.651/0.655 | 0.8183 | 0.9646 | 0.4461 |
| 3 | quick | `blockwise_switch_snr01_uniform` | 0 | 0.5918 | 0.6243 | 0.5149 | 0.683/0.509/0.681 | 0.7861 | 0.9823 | 0.4485 |
| 4 | quick | `blockwise_switch_snr06_triangular` | 0 | 0.5470 | 0.5455 | 0.4589 | 0.842/0.310/0.484 | 0.7648 | 0.8910 | 0.4461 |
| 5 | quick | `single_bw_snr03_uniform` | 0 | 0.4983 | 0.3732 | 0.3815 | 0.458/0.569/0.092 | 0.7775 | 0.8951 | 0.4454 |
| 6 | quick | `single_bw_snr04_uniform` | 0 | 0.5500 | 0.3837 | 0.3808 | 0.551/0.600/0.000 | 0.6381 | 0.4278 | 0.4459 |
| 7 | quick | `single_bw_snr06_triangular` | 0 | 0.5161 | 0.3639 | 0.3749 | 0.432/0.630/0.029 | 0.7738 | 0.9728 | 0.4457 |
| 8 | quick | `single_bw_snr08_triangular` | 0 | 0.4806 | 0.3611 | 0.3741 | 0.482/0.516/0.085 | 0.7698 | 0.9578 | 0.4474 |
| 9 | quick | `single_bw_snr02_uniform` | 0 | 0.3767 | 0.3158 | 0.3125 | 0.304/0.454/0.190 | 0.7888 | 0.9823 | 0.4453 |
| 10 | quick | `single_bw_snr01_uniform` | 0 | 0.3573 | 0.3440 | 0.3098 | 0.329/0.384/0.319 | 0.7920 | 0.9946 | 0.4485 |
| 11 | quick | `single_bw_snr05_triangular` | 0 | 0.3349 | 0.3136 | 0.2897 | 0.269/0.394/0.277 | 0.7711 | 0.9932 | 0.4482 |
| 12 | quick | `single_bw_snr07_triangular` | 0 | 0.2694 | 0.2562 | 0.2146 | 0.118/0.395/0.255 | 0.8397 | 0.9932 | 0.4435 |
