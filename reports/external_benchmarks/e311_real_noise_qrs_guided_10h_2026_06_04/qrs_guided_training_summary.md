# QRS-Guided Real Noise 10h Training Summary

Reference h_bad_rescue_05: acc 0.8229, balanced 0.8177, macro-F1 0.7454.

| rank | mode | variant | return | BUT acc | BUT bal | BUT macro | recalls G/M/B | PTB acc | PTB bad |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 1 | full | `blockwise_switch_snr02_uniform__bad_atten_only_mild` | 0 | 0.6311 | 0.7230 | 0.5487 | 0.895/0.391/0.883 | 0.8647 | 0.9809 |
| 2 | full | `blockwise_switch_snr06_triangular__bad_atten_only_mild` | 0 | 0.5531 | 0.6776 | 0.4993 | 0.671/0.420/0.942 | 0.8470 | 0.9891 |
| 3 | quick | `single_bw_snr03_uniform__nonqrs_medium_bad_guard` | 0 | 0.6045 | 0.5730 | 0.4939 | 0.887/0.387/0.445 | 0.9614 | 0.9891 |
| 4 | quick | `single_bw_snr03_uniform__qrs_soft_boundary` | 0 | 0.5761 | 0.5234 | 0.4732 | 0.788/0.422/0.360 | 0.9255 | 0.9809 |
| 5 | quick | `single_bw_snr02_uniform__medium_detail_qrs_preserved` | 0 | 0.5455 | 0.5869 | 0.4707 | 0.714/0.397/0.650 | 0.9346 | 0.9946 |
| 6 | full | `single_bw_snr02_uniform__medium_detail_qrs_preserved` | 0 | 0.5422 | 0.5757 | 0.4705 | 0.734/0.378/0.616 | 0.9564 | 0.9946 |
| 7 | full | `single_bw_snr03_uniform__good_qrs_cleaner` | 0 | 0.5737 | 0.5036 | 0.4673 | 0.791/0.420/0.299 | 0.9396 | 0.9918 |
| 8 | quick | `single_bw_snr02_uniform__qrs_soft_boundary` | 0 | 0.5653 | 0.5094 | 0.4633 | 0.756/0.429/0.343 | 0.9055 | 0.9932 |
| 9 | quick | `blockwise_switch_snr02_uniform__bad_atten_only_mild` | 0 | 0.5366 | 0.6732 | 0.4596 | 0.850/0.243/0.927 | 0.8442 | 0.9864 |
| 10 | quick | `blockwise_switch_snr06_triangular__bad_atten_only_mild` | 0 | 0.5298 | 0.6683 | 0.4593 | 0.816/0.257/0.932 | 0.8188 | 0.9755 |
| 11 | full | `single_bw_snr02_uniform__bad_atten_only_mild` | 0 | 0.5973 | 0.4581 | 0.4550 | 0.515/0.706/0.153 | 0.8815 | 0.9864 |
| 12 | full | `single_bw_snr02_uniform__qrs_soft_boundary` | 0 | 0.5364 | 0.5441 | 0.4507 | 0.760/0.354/0.518 | 0.9355 | 0.9973 |
| 13 | quick | `single_bw_snr03_uniform__medium_detail_qrs_preserved` | 0 | 0.5442 | 0.5264 | 0.4497 | 0.799/0.344/0.436 | 0.9519 | 0.9850 |
| 14 | quick | `single_bw_snr03_uniform__good_qrs_cleaner` | 0 | 0.5452 | 0.5246 | 0.4494 | 0.820/0.331/0.423 | 0.9214 | 0.9905 |
| 15 | quick | `single_bw_snr03_uniform__bad_atten_only_mild` | 0 | 0.5677 | 0.4817 | 0.4472 | 0.785/0.419/0.241 | 0.8896 | 0.9809 |
| 16 | full | `single_bw_snr08_triangular__good_qrs_cleaner` | 0 | 0.5330 | 0.4839 | 0.4438 | 0.687/0.424/0.341 | 0.9178 | 0.9932 |
| 17 | quick | `single_bw_snr02_uniform__bad_atten_only_mild` | 0 | 0.5181 | 0.4871 | 0.4306 | 0.652/0.420/0.389 | 0.8660 | 0.9877 |
| 18 | full | `single_bw_snr03_uniform__medium_detail_qrs_preserved` | 0 | 0.4949 | 0.5252 | 0.4127 | 0.776/0.259/0.540 | 0.9687 | 0.9905 |
| 19 | full | `single_bw_snr08_triangular__bad_atten_only_mild` | 0 | 0.5050 | 0.4405 | 0.4086 | 0.442/0.576/0.304 | 0.8778 | 0.9905 |
| 20 | full | `single_bw_snr03_uniform__nonqrs_medium_bad_guard` | 0 | 0.5100 | 0.4660 | 0.4053 | 0.843/0.256/0.299 | 0.9700 | 0.9850 |
| 21 | full | `single_bw_snr03_uniform__qrs_soft_boundary` | 0 | 0.4876 | 0.5215 | 0.4036 | 0.780/0.242/0.543 | 0.9455 | 0.9905 |
| 22 | quick | `single_bw_snr08_triangular__good_qrs_cleaner` | 0 | 0.4525 | 0.5064 | 0.3992 | 0.493/0.404/0.623 | 0.8955 | 0.9946 |
| 23 | full | `single_bw_snr03_uniform__bad_atten_only_mild` | 0 | 0.4553 | 0.4838 | 0.3807 | 0.714/0.239/0.499 | 0.8996 | 0.9877 |
| 24 | quick | `single_bw_snr08_triangular__bad_atten_only_mild` | 0 | 0.4478 | 0.4121 | 0.3693 | 0.578/0.354/0.304 | 0.8629 | 0.9768 |
