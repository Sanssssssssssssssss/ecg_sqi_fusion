# E3.11 Mainline Grid

Mainline decision: E3.11f-style visual data replaces E3.10 as the target line.
The source clean pool uses strict PTB-XL filtering: any baseline/static/burst/electrode annotation in any lead is rejected.
E3.10 remains only a historical reference, not an optimization target.

Current historical references:

- E3.10 best single visual: `0.9402`
- E3.10 calibration/ensemble diagnostic: `0.9419`
- Previous E3.11f best single visual: `0.9381`
- Previous E3.11f ensemble diagnostic: `0.9402`

Stage-1 goal:

- find an E3.11f single model at or above `0.94`
- screen whether relaxed morphology or wider SNR makes the task more learnable
- prune any branch that is clearly worse before expanding the grid

## Data Variants

| Variant | Meaning | Class Counts G/M/B | measured SNR Mean G/M/B | smooth Morph Mean G/M/B | global Noise Mean G/M/B | SQI-SVM | SQI-MLP |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `e311f_lite_e310_morph` | E3.11f lite SNR + E3.10 morphology | train: 3645/3645/3645; val: 728/728/728; test: 734/734/734 | 12.243/9.069/6.810 | 0.074/0.310/0.402 | 0.245/0.353/0.457 | 0.5263 | 0.5118 |
| `e311h_lite_relaxed_morph` | E3.11h lite SNR + relaxed morphology | train: 3987/3987/3987; val: 795/795/795; test: 799/799/799 | 12.240/9.101/6.914 | 0.073/0.296/0.395 | 0.245/0.351/0.451 | 0.5294 | 0.5048 |
| `e311i_wide_relaxed_morph` | E3.11i wide SNR + relaxed morphology | train: 3998/3998/3998; val: 799/799/799; test: 800/800/800 | 13.944/7.958/4.136 | 0.067/0.310/0.572 | 0.201/0.401/0.623 | 0.5646 | 0.5563 |

## Transformer Results

### e311f_lite_e310_morph

E3.11f lite SNR + E3.10 morphology.

| Rank | Run | Test Acc | Summary |
| ---: | --- | ---: | --- |
| 1 | `e311f_lite_e310_morph_f05_lr5_pos` | 0.9391 | acc=0.9391, rec=0.9128/0.9346/0.9700, pool=cls, pos=True, snr=0.05, ord=False, noise=False, den=0.0, lr=5e-05, drop=0.1 |
| 2 | `e311f_lite_e310_morph_f23_seed1_pos` | 0.9391 | acc=0.9391, rec=0.9264/0.9196/0.9714, pool=cls, pos=True, snr=0.05, ord=False, noise=False, den=0.0, lr=3e-05, drop=0.1 |
| 3 | `e311f_lite_e310_morph_f16_good104_pos` | 0.9387 | acc=0.9387, rec=0.9251/0.9264/0.9646, pool=cls, pos=True, snr=0.05, ord=False, noise=False, den=0.0, lr=3e-05, drop=0.1 |
| 4 | `e311f_lite_e310_morph_f11_noise_pos` | 0.9382 | acc=0.9382, rec=0.9319/0.9060/0.9768, pool=cls, pos=True, snr=0.05, ord=False, noise=True, den=0.0, lr=3e-05, drop=0.1 |
| 5 | `e311f_lite_e310_morph_f03_clsmean_pos` | 0.9378 | acc=0.9378, rec=0.9169/0.9387/0.9578, pool=cls_mean, pos=True, snr=0.05, ord=False, noise=False, den=0.0, lr=3e-05, drop=0.1 |
| 6 | `e311f_lite_e310_morph_f13_lateden_pos` | 0.9378 | acc=0.9378, rec=0.9346/0.9114/0.9673, pool=cls, pos=True, snr=0.05, ord=False, noise=False, den=40.0, lr=3e-05, drop=0.1 |
| 7 | `e311f_lite_e310_morph_f17_med104_pos` | 0.9378 | acc=0.9378, rec=0.9169/0.9223/0.9741, pool=cls, pos=True, snr=0.05, ord=False, noise=False, den=0.0, lr=3e-05, drop=0.1 |
| 8 | `e311f_lite_e310_morph_f01_pos` | 0.9373 | acc=0.9373, rec=0.9128/0.9237/0.9755, pool=cls, pos=True, snr=0.05, ord=False, noise=False, den=0.0, lr=3e-05, drop=0.1 |
| 9 | `e311f_lite_e310_morph_f09_snr010_pos` | 0.9369 | acc=0.9369, rec=0.9087/0.9264/0.9755, pool=cls, pos=True, snr=0.1, ord=False, noise=False, den=0.0, lr=3e-05, drop=0.1 |
| 10 | `e311f_lite_e310_morph_f21_wd005_pos` | 0.9369 | acc=0.9369, rec=0.9142/0.9223/0.9741, pool=cls, pos=True, snr=0.05, ord=False, noise=False, den=0.0, lr=3e-05, drop=0.1 |
| 11 | `e311f_lite_e310_morph_f20_wd001_pos` | 0.9364 | acc=0.9364, rec=0.9128/0.9223/0.9741, pool=cls, pos=True, snr=0.05, ord=False, noise=False, den=0.0, lr=3e-05, drop=0.1 |
| 12 | `e311f_lite_e310_morph_f06_drop015_pos` | 0.9360 | acc=0.9360, rec=0.9183/0.9169/0.9728, pool=cls, pos=True, snr=0.05, ord=False, noise=False, den=0.0, lr=3e-05, drop=0.15 |

### e311h_lite_relaxed_morph

E3.11h lite SNR + relaxed morphology.

| Rank | Run | Test Acc | Summary |
| ---: | --- | ---: | --- |
| 1 | `e311h_lite_relaxed_morph_h03_snr010_pos` | 0.9103 | acc=0.9103, rec=0.9287/0.8711/0.9312, pool=cls, pos=True, snr=0.1, ord=False, noise=False, den=0.0, lr=3e-05, drop=0.1 |
| 2 | `e311h_lite_relaxed_morph_h01_pos` | 0.9095 | acc=0.9095, rec=0.9337/0.8536/0.9412, pool=cls, pos=True, snr=0.05, ord=False, noise=False, den=0.0, lr=3e-05, drop=0.1 |
| 3 | `e311h_lite_relaxed_morph_h00_anchor` | 0.9070 | acc=0.9070, rec=0.9124/0.8836/0.9249, pool=cls, pos=False, snr=0.05, ord=False, noise=False, den=0.0, lr=3e-05, drop=0.1 |
| 4 | `e311h_lite_relaxed_morph_h04_ord_pos` | 0.9045 | acc=0.9045, rec=0.9262/0.8423/0.9449, pool=cls, pos=True, snr=0.05, ord=True, noise=False, den=0.0, lr=3e-05, drop=0.1 |
| 5 | `e311h_lite_relaxed_morph_h02_clsmean_pos` | 0.9028 | acc=0.9028, rec=0.9174/0.8523/0.9387, pool=cls_mean, pos=True, snr=0.05, ord=False, noise=False, den=0.0, lr=3e-05, drop=0.1 |

### e311i_wide_relaxed_morph

E3.11i wide SNR + relaxed morphology.

| Rank | Run | Test Acc | Summary |
| ---: | --- | ---: | --- |
|  | pending |  |  |

## Current Best

| Rank | Variant | Run | Test Acc | Summary |
| ---: | --- | --- | ---: | --- |
| 1 | `e311f_lite_e310_morph` | `e311f_lite_e310_morph_f05_lr5_pos` | 0.9391 | acc=0.9391, rec=0.9128/0.9346/0.9700, pool=cls, pos=True, snr=0.05, ord=False, noise=False, den=0.0, lr=5e-05, drop=0.1 |
| 2 | `e311f_lite_e310_morph` | `e311f_lite_e310_morph_f23_seed1_pos` | 0.9391 | acc=0.9391, rec=0.9264/0.9196/0.9714, pool=cls, pos=True, snr=0.05, ord=False, noise=False, den=0.0, lr=3e-05, drop=0.1 |
| 3 | `e311f_lite_e310_morph` | `e311f_lite_e310_morph_f16_good104_pos` | 0.9387 | acc=0.9387, rec=0.9251/0.9264/0.9646, pool=cls, pos=True, snr=0.05, ord=False, noise=False, den=0.0, lr=3e-05, drop=0.1 |
| 4 | `e311f_lite_e310_morph` | `e311f_lite_e310_morph_f11_noise_pos` | 0.9382 | acc=0.9382, rec=0.9319/0.9060/0.9768, pool=cls, pos=True, snr=0.05, ord=False, noise=True, den=0.0, lr=3e-05, drop=0.1 |
| 5 | `e311f_lite_e310_morph` | `e311f_lite_e310_morph_f03_clsmean_pos` | 0.9378 | acc=0.9378, rec=0.9169/0.9387/0.9578, pool=cls_mean, pos=True, snr=0.05, ord=False, noise=False, den=0.0, lr=3e-05, drop=0.1 |
| 6 | `e311f_lite_e310_morph` | `e311f_lite_e310_morph_f13_lateden_pos` | 0.9378 | acc=0.9378, rec=0.9346/0.9114/0.9673, pool=cls, pos=True, snr=0.05, ord=False, noise=False, den=40.0, lr=3e-05, drop=0.1 |
| 7 | `e311f_lite_e310_morph` | `e311f_lite_e310_morph_f17_med104_pos` | 0.9378 | acc=0.9378, rec=0.9169/0.9223/0.9741, pool=cls, pos=True, snr=0.05, ord=False, noise=False, den=0.0, lr=3e-05, drop=0.1 |
| 8 | `e311f_lite_e310_morph` | `e311f_lite_e310_morph_f01_pos` | 0.9373 | acc=0.9373, rec=0.9128/0.9237/0.9755, pool=cls, pos=True, snr=0.05, ord=False, noise=False, den=0.0, lr=3e-05, drop=0.1 |
| 9 | `e311f_lite_e310_morph` | `e311f_lite_e310_morph_f09_snr010_pos` | 0.9369 | acc=0.9369, rec=0.9087/0.9264/0.9755, pool=cls, pos=True, snr=0.1, ord=False, noise=False, den=0.0, lr=3e-05, drop=0.1 |
| 10 | `e311f_lite_e310_morph` | `e311f_lite_e310_morph_f21_wd005_pos` | 0.9369 | acc=0.9369, rec=0.9142/0.9223/0.9741, pool=cls, pos=True, snr=0.05, ord=False, noise=False, den=0.0, lr=3e-05, drop=0.1 |
| 11 | `e311f_lite_e310_morph` | `e311f_lite_e310_morph_f20_wd001_pos` | 0.9364 | acc=0.9364, rec=0.9128/0.9223/0.9741, pool=cls, pos=True, snr=0.05, ord=False, noise=False, den=0.0, lr=3e-05, drop=0.1 |
| 12 | `e311f_lite_e310_morph` | `e311f_lite_e310_morph_f06_drop015_pos` | 0.9360 | acc=0.9360, rec=0.9183/0.9169/0.9728, pool=cls, pos=True, snr=0.05, ord=False, noise=False, den=0.0, lr=3e-05, drop=0.15 |
| 13 | `e311h_lite_relaxed_morph` | `e311h_lite_relaxed_morph_h03_snr010_pos` | 0.9103 | acc=0.9103, rec=0.9287/0.8711/0.9312, pool=cls, pos=True, snr=0.1, ord=False, noise=False, den=0.0, lr=3e-05, drop=0.1 |
| 14 | `e311h_lite_relaxed_morph` | `e311h_lite_relaxed_morph_h01_pos` | 0.9095 | acc=0.9095, rec=0.9337/0.8536/0.9412, pool=cls, pos=True, snr=0.05, ord=False, noise=False, den=0.0, lr=3e-05, drop=0.1 |
| 15 | `e311h_lite_relaxed_morph` | `e311h_lite_relaxed_morph_h00_anchor` | 0.9070 | acc=0.9070, rec=0.9124/0.8836/0.9249, pool=cls, pos=False, snr=0.05, ord=False, noise=False, den=0.0, lr=3e-05, drop=0.1 |

## Stage Decisions

- Keep `e311f_lite_e310_morph` as the main E3.11 data line: it is the only branch close to the historical E3.10 visual result.
- Prune `e311h_lite_relaxed_morph`: the relaxed-morph screen stayed around `0.90-0.91`, far below E3.11f.
- Prune `e311i_wide_relaxed_morph`: it has higher SQI baselines and early validation was clearly worse than E3.11f, so the wide-SNR branch is diagnostic only.
- Round 2 should focus on `e311f_lite_e310_morph` with positional CLS models near `lr=3e-5..6e-5`, SNR-head weight, light class weighting, and seed robustness.

## Pruning Rules

- If a variant's first 3-6 runs stay below `0.90`, stop expanding it.
- If relaxed morphology beats E3.11f by `>=0.01`, expand that data branch.
- If wide SNR is high but SQI baselines are also high, keep it as visual/diagnostic rather than the main benchmark.
- If positional embedding helps the top E3.11f runs, keep it in round 2; otherwise remove it.
- If denoise-aware runs lose more than `0.01` versus their paired classifier-only run, remove denoise from the next grid.
