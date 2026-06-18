# N7200 Geometry Disagreement

## Model Endpoints

- `old_best`: acc 0.9364, macro-F1 0.9436, recall good/medium/bad 0.9314/0.9221/0.9706
- `good_heavy`: acc 0.9273, macro-F1 0.9355, recall good/medium/bad 0.9810/0.8489/0.9709
- `medium_heavy`: acc 0.8310, macro-F1 0.8512, recall good/medium/bad 0.6521/0.9307/0.9706

## Disagreement Counts

- `other`: 15107
- `good_lost_by_mediumheavy`: 2397
- `medium_rescued_by_mediumheavy`: 808
- `good_rescued_by_goodheavy`: 99
- `medium_lost_by_goodheavy`: 73

## Top Good-Heavy Separator Features

- `pc1` KS 1.000, median rescued/lost -4.176/-0.7822
- `pc3` KS 0.952, median rescued/lost -2.389/2.663
- `flatline_ratio` KS 0.912, median rescued/lost 0.3155/0.1193
- `qrs_visibility` KS 0.835, median rescued/lost 0.5627/0.1918
- `non_qrs_diff_p95` KS 0.823, median rescued/lost 0.03696/0.07646
- `knn_label_purity` KS 0.671, median rescued/lost 1/0.8667

## Top Medium-Heavy Separator Features

- `pc1` KS 1.000, median medium-rescued/good-lost -0.4167/-3.62
- `flatline_ratio` KS 0.936, median medium-rescued/good-lost 0.08567/0.2242
- `pc3` KS 0.907, median medium-rescued/good-lost 2.899/-0.4247
- `non_qrs_diff_p95` KS 0.899, median medium-rescued/good-lost 0.13/0.06453
- `qrs_visibility` KS 0.873, median medium-rescued/good-lost 0.3889/0.6238
- `template_corr` KS 0.655, median medium-rescued/good-lost 0.5658/0.6652
