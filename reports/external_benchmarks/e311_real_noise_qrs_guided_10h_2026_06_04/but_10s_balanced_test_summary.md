# BUT 10s Balanced-Test Evaluation

This is an evaluation-only balanced subset of the formal BUT 10s P1 test split. It keeps validation-only calibration unchanged and samples the test split to equal class counts.

| rank | mode | variant | balanced acc | balanced macro | recalls G/M/B | original acc | original macro | original recalls G/M/B |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: | --- |
| 1 | full | `blockwise_switch_snr02_uniform__bad_atten_only_mild` | 0.7072 | 0.6820 | 0.878/0.360/0.883 | 0.6311 | 0.5487 | 0.895/0.391/0.883 |
| 2 | full | `blockwise_switch_snr06_triangular__bad_atten_only_mild` | 0.6496 | 0.6294 | 0.647/0.360/0.942 | 0.5531 | 0.4993 | 0.671/0.420/0.942 |
| 3 | quick | `blockwise_switch_snr02_uniform__bad_atten_only_mild` | 0.6569 | 0.6059 | 0.839/0.204/0.927 | 0.5366 | 0.4596 | 0.850/0.243/0.927 |
| 4 | quick | `blockwise_switch_snr06_triangular__bad_atten_only_mild` | 0.6496 | 0.6056 | 0.788/0.229/0.932 | 0.5298 | 0.4593 | 0.816/0.257/0.932 |
| 5 | quick | `single_bw_snr02_uniform__medium_detail_qrs_preserved` | 0.5758 | 0.5713 | 0.681/0.397/0.650 | 0.5455 | 0.4707 | 0.714/0.397/0.650 |
| 6 | full | `single_bw_snr02_uniform__medium_detail_qrs_preserved` | 0.5596 | 0.5586 | 0.708/0.355/0.616 | 0.5422 | 0.4705 | 0.734/0.378/0.616 |
| 7 | quick | `single_bw_snr03_uniform__nonqrs_medium_bad_guard` | 0.5661 | 0.5468 | 0.878/0.375/0.445 | 0.6045 | 0.4939 | 0.887/0.387/0.445 |
| 8 | full | `single_bw_snr02_uniform__qrs_soft_boundary` | 0.5207 | 0.5128 | 0.723/0.321/0.518 | 0.5364 | 0.4507 | 0.760/0.354/0.518 |
| 9 | quick | `single_bw_snr03_uniform__medium_detail_qrs_preserved` | 0.5182 | 0.5091 | 0.769/0.350/0.436 | 0.5442 | 0.4497 | 0.799/0.344/0.436 |
| 10 | quick | `single_bw_snr03_uniform__qrs_soft_boundary` | 0.5150 | 0.5042 | 0.757/0.428/0.360 | 0.5761 | 0.4732 | 0.788/0.422/0.360 |
| 11 | quick | `single_bw_snr03_uniform__good_qrs_cleaner` | 0.5150 | 0.4996 | 0.796/0.326/0.423 | 0.5452 | 0.4494 | 0.820/0.331/0.423 |
| 12 | full | `single_bw_snr03_uniform__medium_detail_qrs_preserved` | 0.5109 | 0.4987 | 0.745/0.248/0.540 | 0.4949 | 0.4127 | 0.776/0.259/0.540 |
| 13 | quick | `single_bw_snr02_uniform__qrs_soft_boundary` | 0.5061 | 0.4957 | 0.740/0.436/0.343 | 0.5653 | 0.4633 | 0.756/0.429/0.343 |
| 14 | full | `single_bw_snr03_uniform__qrs_soft_boundary` | 0.5101 | 0.4870 | 0.762/0.226/0.543 | 0.4876 | 0.4036 | 0.780/0.242/0.543 |
| 15 | quick | `single_bw_snr08_triangular__good_qrs_cleaner` | 0.4915 | 0.4865 | 0.489/0.363/0.623 | 0.4525 | 0.3992 | 0.493/0.404/0.623 |
| 16 | full | `single_bw_snr03_uniform__good_qrs_cleaner` | 0.4996 | 0.4839 | 0.764/0.436/0.299 | 0.5737 | 0.4673 | 0.791/0.420/0.299 |
| 17 | full | `single_bw_snr08_triangular__good_qrs_cleaner` | 0.4745 | 0.4677 | 0.676/0.406/0.341 | 0.5330 | 0.4438 | 0.687/0.424/0.341 |
| 18 | quick | `single_bw_snr02_uniform__bad_atten_only_mild` | 0.4680 | 0.4634 | 0.628/0.387/0.389 | 0.5181 | 0.4306 | 0.652/0.420/0.389 |
| 19 | full | `single_bw_snr03_uniform__bad_atten_only_mild` | 0.4745 | 0.4546 | 0.701/0.224/0.499 | 0.4553 | 0.3807 | 0.714/0.239/0.499 |
| 20 | quick | `single_bw_snr03_uniform__bad_atten_only_mild` | 0.4769 | 0.4528 | 0.771/0.418/0.241 | 0.5677 | 0.4472 | 0.785/0.419/0.241 |
| 21 | full | `single_bw_snr03_uniform__nonqrs_medium_bad_guard` | 0.4663 | 0.4376 | 0.835/0.265/0.299 | 0.5100 | 0.4053 | 0.843/0.256/0.299 |
| 22 | full | `single_bw_snr08_triangular__bad_atten_only_mild` | 0.4404 | 0.4356 | 0.470/0.547/0.304 | 0.5050 | 0.4086 | 0.442/0.576/0.304 |
| 23 | full | `single_bw_snr02_uniform__bad_atten_only_mild` | 0.4558 | 0.4304 | 0.513/0.701/0.153 | 0.5973 | 0.4550 | 0.515/0.706/0.153 |
| 24 | quick | `single_bw_snr08_triangular__bad_atten_only_mild` | 0.4031 | 0.3976 | 0.567/0.338/0.304 | 0.4478 | 0.3693 | 0.578/0.354/0.304 |
