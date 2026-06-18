# Medium-Cluster Real Noise Training Summary

Reference qrs balanced best raw: macro-F1 0.6997, recalls [0.864, 0.436, 0.844].
Reference h_bad_rescue_05 original BUT: macro-F1 0.7454, recalls [0.887, 0.773, 0.793].

| rank | mode | variant | return | bal cal macro | bal raw macro | raw recalls G/M/B | cal recalls G/M/B | orig macro | PTB acc | PTB bad |
| --- | --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |
| 1 | quick | `blockwise_snr06_triangular__m08_medium_boundary_flat__cw1p00_1p75_1p55` | 0 | 0.2749 | 0.1503 | 0.000/0.041/0.710 | 0.467/0.489/0.027 | 0.3349 | 0.4688 | 1.0000 |
| 2 | quick | `blockwise_snr06_triangular__m08_medium_boundary_flat__cw1p00_1p55_1p70` | 0 | 0.2517 | 0.1414 | 0.000/0.024/0.710 | 0.389/0.555/0.002 | 0.3104 | 0.4688 | 1.0000 |
| 3 | quick | `blockwise_snr06_triangular__m08_medium_boundary_flat__cw1p00_1p90_1p45` | 4294967295 | 0.0000 | 0.0000 | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 0.0000 | 0.0000 | 0.0000 |
| 4 | quick | `blockwise_snr06_triangular__m07_medium_raw_logit_probe__cw1p00_1p55_1p70` | 4294967295 | 0.0000 | 0.0000 | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 0.0000 | 0.0000 | 0.0000 |
| 5 | quick | `blockwise_snr06_triangular__m07_medium_raw_logit_probe__cw1p00_1p75_1p55` | 4294967295 | 0.0000 | 0.0000 | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 0.0000 | 0.0000 | 0.0000 |
