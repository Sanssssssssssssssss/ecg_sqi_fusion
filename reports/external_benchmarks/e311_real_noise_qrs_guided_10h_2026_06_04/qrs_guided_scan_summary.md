# QRS-Guided Real NSTDB Noise Scan

This scan keeps real NSTDB EM/MA/BW as the only noise source and applies small QRS-aware noise placement/gain changes.

| rank | variant | base | profile | score | morph | medium | SQI | domain |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `blockwise_switch_snr06_triangular__bad_atten_only_mild` | `blockwise_switch_snr06_triangular` | bad_atten_only_mild | 0.4332 | 0.4243 | 0.2565 | 0.4385 | 0.9123 |
| 2 | `single_bw_snr03_uniform__medium_detail_qrs_preserved` | `single_bw_snr03_uniform` | medium_detail_qrs_preserved | 0.4333 | 0.4318 | 0.2579 | 0.4325 | 0.8806 |
| 3 | `single_bw_snr02_uniform__bad_atten_only_mild` | `single_bw_snr02_uniform` | bad_atten_only_mild | 0.4340 | 0.4282 | 0.2578 | 0.4306 | 0.9087 |
| 4 | `single_bw_snr02_uniform__medium_detail_qrs_preserved` | `single_bw_snr02_uniform` | medium_detail_qrs_preserved | 0.4345 | 0.4312 | 0.2555 | 0.4286 | 0.9075 |
| 5 | `single_bw_snr03_uniform__qrs_soft_boundary` | `single_bw_snr03_uniform` | qrs_soft_boundary | 0.4353 | 0.4330 | 0.2606 | 0.4325 | 0.8873 |
| 6 | `single_bw_snr02_uniform__qrs_soft_boundary` | `single_bw_snr02_uniform` | qrs_soft_boundary | 0.4360 | 0.4323 | 0.2561 | 0.4345 | 0.9063 |
| 7 | `single_bw_snr08_triangular__good_qrs_cleaner` | `single_bw_snr08_triangular` | good_qrs_cleaner | 0.4363 | 0.4352 | 0.2573 | 0.4306 | 0.8976 |
| 8 | `single_bw_snr03_uniform__nonqrs_medium_bad_guard` | `single_bw_snr03_uniform` | nonqrs_medium_bad_guard | 0.4363 | 0.4327 | 0.2605 | 0.4405 | 0.8873 |
| 9 | `single_bw_snr08_triangular__bad_atten_only_mild` | `single_bw_snr08_triangular` | bad_atten_only_mild | 0.4364 | 0.4327 | 0.2610 | 0.4325 | 0.8988 |
| 10 | `single_bw_snr03_uniform__bad_atten_only_mild` | `single_bw_snr03_uniform` | bad_atten_only_mild | 0.4366 | 0.4302 | 0.2614 | 0.4405 | 0.9008 |
| 11 | `single_bw_snr03_uniform__good_qrs_cleaner` | `single_bw_snr03_uniform` | good_qrs_cleaner | 0.4366 | 0.4331 | 0.2613 | 0.4385 | 0.8897 |
| 12 | `blockwise_switch_snr02_uniform__bad_atten_only_mild` | `blockwise_switch_snr02_uniform` | bad_atten_only_mild | 0.4366 | 0.4236 | 0.2570 | 0.4385 | 0.9484 |
| 13 | `blockwise_switch_snr06_triangular__medium_detail_qrs_preserved` | `blockwise_switch_snr06_triangular` | medium_detail_qrs_preserved | 0.4367 | 0.4252 | 0.2586 | 0.4405 | 0.9337 |
| 14 | `single_bw_snr02_uniform__good_qrs_cleaner` | `single_bw_snr02_uniform` | good_qrs_cleaner | 0.4367 | 0.4320 | 0.2559 | 0.4345 | 0.9155 |
| 15 | `blockwise_switch_snr06_triangular__good_qrs_cleaner` | `blockwise_switch_snr06_triangular` | good_qrs_cleaner | 0.4369 | 0.4262 | 0.2580 | 0.4385 | 0.9349 |
| 16 | `single_bw_snr06_triangular__good_qrs_cleaner` | `single_bw_snr06_triangular` | good_qrs_cleaner | 0.4370 | 0.4324 | 0.2578 | 0.4444 | 0.8964 |
| 17 | `single_bw_snr03_uniform__qrs_boundary_balanced` | `single_bw_snr03_uniform` | qrs_boundary_balanced | 0.4370 | 0.4331 | 0.2606 | 0.4425 | 0.8897 |
| 18 | `single_bw_snr06_triangular__bad_qrs_focus_light` | `single_bw_snr06_triangular` | bad_qrs_focus_light | 0.4371 | 0.4330 | 0.2573 | 0.4385 | 0.9044 |
| 19 | `blockwise_switch_snr06_triangular__qrs_soft_boundary` | `blockwise_switch_snr06_triangular` | qrs_soft_boundary | 0.4371 | 0.4254 | 0.2584 | 0.4425 | 0.9337 |
| 20 | `single_bw_snr03_uniform__bad_qrs_focus_light` | `single_bw_snr03_uniform` | bad_qrs_focus_light | 0.4375 | 0.4343 | 0.2626 | 0.4345 | 0.8952 |
| 21 | `blockwise_switch_snr06_triangular__nonqrs_medium_bad_guard` | `blockwise_switch_snr06_triangular` | nonqrs_medium_bad_guard | 0.4375 | 0.4266 | 0.2591 | 0.4464 | 0.9246 |
| 22 | `blockwise_switch_snr06_triangular__qrs_boundary_balanced` | `blockwise_switch_snr06_triangular` | qrs_boundary_balanced | 0.4375 | 0.4272 | 0.2584 | 0.4484 | 0.9202 |
| 23 | `single_bw_snr02_uniform__nonqrs_medium_bad_guard` | `single_bw_snr02_uniform` | nonqrs_medium_bad_guard | 0.4377 | 0.4326 | 0.2578 | 0.4365 | 0.9143 |
| 24 | `single_bw_snr08_triangular__qrs_soft_boundary` | `single_bw_snr08_triangular` | qrs_soft_boundary | 0.4378 | 0.4357 | 0.2575 | 0.4405 | 0.8952 |
| 25 | `single_bw_snr06_triangular__bad_qrs_focus_strong` | `single_bw_snr06_triangular` | bad_qrs_focus_strong | 0.4378 | 0.4354 | 0.2584 | 0.4385 | 0.8976 |
| 26 | `blockwise_switch_snr02_uniform__medium_detail_qrs_preserved` | `blockwise_switch_snr02_uniform` | medium_detail_qrs_preserved | 0.4378 | 0.4256 | 0.2597 | 0.4306 | 0.9552 |
| 27 | `single_bw_snr02_uniform__qrs_boundary_balanced` | `single_bw_snr02_uniform` | qrs_boundary_balanced | 0.4378 | 0.4331 | 0.2559 | 0.4385 | 0.9155 |
| 28 | `single_bw_snr06_triangular__medium_detail_qrs_preserved` | `single_bw_snr06_triangular` | medium_detail_qrs_preserved | 0.4379 | 0.4315 | 0.2596 | 0.4504 | 0.8964 |
| 29 | `single_bw_snr06_triangular__nonqrs_medium_bad_guard` | `single_bw_snr06_triangular` | nonqrs_medium_bad_guard | 0.4379 | 0.4321 | 0.2589 | 0.4405 | 0.9099 |
| 30 | `single_bw_snr06_triangular__qrs_boundary_balanced` | `single_bw_snr06_triangular` | qrs_boundary_balanced | 0.4379 | 0.4319 | 0.2582 | 0.4464 | 0.9044 |
