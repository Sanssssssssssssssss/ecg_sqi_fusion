# E3.11 Mainline Grid

Mainline decision: E3.11f-style visual data replaces E3.10 as the target line.
The source clean pool uses strict PTB-XL filtering: any baseline/static/burst/electrode annotation in any lead is rejected.
E3.10 remains only a historical reference, not an optimization target.

Current historical references:

- E3.10 best single visual: `0.9402`
- E3.10 calibration/ensemble diagnostic: `0.9419`
- Previous E3.11f best single visual: `0.9381`
- Previous E3.11f ensemble diagnostic: `0.9402`

Sweep goal:

- keep E3.11f as the visual mainline and find a stable single model at or above `0.94`
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
| 1 | `e311f_lite_e310_morph_hc22_mask010_lr625_s1` | 0.9505 | acc=0.9505, rec=0.9278/0.9496/0.9741, seed=1, pool=cls, pos=True, in=raw, snr=0.05, ord=False, noise=False, mask=0.01, rank=, den=0.0, lvl=0.0, lr=6.25e-05, drop=0.1 |
| 2 | `e311f_lite_e310_morph_hc2_m010_lr625_s1` | 0.9505 | acc=0.9505, rec=0.9278/0.9496/0.9741, seed=1, pool=cls, pos=True, in=raw, snr=0.05, ord=False, noise=False, mask=0.01, rank=, den=0.0, lvl=0.0, lr=6.25e-05, drop=0.1 |
| 3 | `e311f_lite_e310_morph_hc19_mask010_lr575_s1` | 0.9491 | acc=0.9491, rec=0.9305/0.9387/0.9782, seed=1, pool=cls, pos=True, in=raw, snr=0.05, ord=False, noise=False, mask=0.01, rank=, den=0.0, lvl=0.0, lr=5.75e-05, drop=0.1 |
| 4 | `e311f_lite_e310_morph_hc2_m010_lr575_s1` | 0.9491 | acc=0.9491, rec=0.9305/0.9387/0.9782, seed=1, pool=cls, pos=True, in=raw, snr=0.05, ord=False, noise=False, mask=0.01, rank=, den=0.0, lvl=0.0, lr=5.75e-05, drop=0.1 |
| 5 | `e311f_lite_e310_morph_hc03_base_lr625_s1` | 0.9464 | acc=0.9464, rec=0.9292/0.9346/0.9755, seed=1, pool=cls, pos=True, in=raw, snr=0.05, ord=False, noise=False, mask=False, rank=, den=0.0, lvl=0.0, lr=6.25e-05, drop=0.1 |
| 6 | `e311f_lite_e310_morph_r3_lr625_seed1` | 0.9464 | acc=0.9464, rec=0.9292/0.9346/0.9755, seed=1, pool=cls, pos=True, in=raw, snr=0.05, ord=False, noise=False, mask=False, rank=, den=0.0, lvl=0.0, lr=6.25e-05, drop=0.1 |
| 7 | `e311f_lite_e310_morph_hc12_ord005_lr575_s6` | 0.9455 | acc=0.9455, rec=0.9373/0.9278/0.9714, seed=6, pool=cls, pos=True, in=raw, snr=0.05, ord=True, noise=False, mask=False, rank=, den=0.0, lvl=0.0, lr=5.75e-05, drop=0.1 |
| 8 | `e311f_lite_e310_morph_hc18_mask005_lr575_s1` | 0.9455 | acc=0.9455, rec=0.9278/0.9455/0.9632, seed=1, pool=cls, pos=True, in=raw, snr=0.05, ord=False, noise=False, mask=0.005, rank=, den=0.0, lvl=0.0, lr=5.75e-05, drop=0.1 |
| 9 | `e311f_lite_e310_morph_hc01_base_lr575_s1` | 0.9450 | acc=0.9450, rec=0.9278/0.9319/0.9755, seed=1, pool=cls, pos=True, in=raw, snr=0.05, ord=False, noise=False, mask=False, rank=, den=0.0, lvl=0.0, lr=5.75e-05, drop=0.1 |
| 10 | `e311f_lite_e310_morph_hc04_base_lr625_s6` | 0.9450 | acc=0.9450, rec=0.9237/0.9360/0.9755, seed=6, pool=cls, pos=True, in=raw, snr=0.05, ord=False, noise=False, mask=False, rank=, den=0.0, lvl=0.0, lr=6.25e-05, drop=0.1 |
| 11 | `e311f_lite_e310_morph_hc14_noise005_lr575_s1` | 0.9450 | acc=0.9450, rec=0.9305/0.9292/0.9755, seed=1, pool=cls, pos=True, in=raw, snr=0.05, ord=False, noise=True, mask=False, rank=, den=0.0, lvl=0.0, lr=5.75e-05, drop=0.1 |
| 12 | `e311f_lite_e310_morph_hc2_m010_lr625_s6` | 0.9450 | acc=0.9450, rec=0.9346/0.9360/0.9646, seed=6, pool=cls, pos=True, in=raw, snr=0.05, ord=False, noise=False, mask=0.01, rank=, den=0.0, lvl=0.0, lr=6.25e-05, drop=0.1 |

### e311h_lite_relaxed_morph

E3.11h lite SNR + relaxed morphology.

| Rank | Run | Test Acc | Summary |
| ---: | --- | ---: | --- |
| 1 | `e311h_lite_relaxed_morph_h03_snr010_pos` | 0.9103 | acc=0.9103, rec=0.9287/0.8711/0.9312, seed=0, pool=cls, pos=True, in=raw, snr=0.1, ord=False, noise=False, mask=False, rank=, den=0.0, lvl=0.0, lr=3e-05, drop=0.1 |
| 2 | `e311h_lite_relaxed_morph_h01_pos` | 0.9095 | acc=0.9095, rec=0.9337/0.8536/0.9412, seed=0, pool=cls, pos=True, in=raw, snr=0.05, ord=False, noise=False, mask=False, rank=, den=0.0, lvl=0.0, lr=3e-05, drop=0.1 |
| 3 | `e311h_lite_relaxed_morph_h00_anchor` | 0.9070 | acc=0.9070, rec=0.9124/0.8836/0.9249, seed=0, pool=cls, pos=False, in=raw, snr=0.05, ord=False, noise=False, mask=False, rank=, den=0.0, lvl=0.0, lr=3e-05, drop=0.1 |
| 4 | `e311h_lite_relaxed_morph_h05_lateden_pos` | 0.9053 | acc=0.9053, rec=0.9249/0.8611/0.9299, seed=0, pool=cls, pos=True, in=raw, snr=0.05, ord=False, noise=False, mask=False, rank=, den=40.0, lvl=1.0, lr=3e-05, drop=0.1 |
| 5 | `e311h_lite_relaxed_morph_h04_ord_pos` | 0.9045 | acc=0.9045, rec=0.9262/0.8423/0.9449, seed=0, pool=cls, pos=True, in=raw, snr=0.05, ord=True, noise=False, mask=False, rank=, den=0.0, lvl=0.0, lr=3e-05, drop=0.1 |
| 6 | `e311h_lite_relaxed_morph_h02_clsmean_pos` | 0.9028 | acc=0.9028, rec=0.9174/0.8523/0.9387, seed=0, pool=cls_mean, pos=True, in=raw, snr=0.05, ord=False, noise=False, mask=False, rank=, den=0.0, lvl=0.0, lr=3e-05, drop=0.1 |

### e311i_wide_relaxed_morph

E3.11i wide SNR + relaxed morphology.

| Rank | Run | Test Acc | Summary |
| ---: | --- | ---: | --- |
|  | pending |  |  |

## Current Best

| Rank | Variant | Run | Test Acc | Summary |
| ---: | --- | --- | ---: | --- |
| 1 | `e311f_lite_e310_morph` | `e311f_lite_e310_morph_hc22_mask010_lr625_s1` | 0.9505 | acc=0.9505, rec=0.9278/0.9496/0.9741, seed=1, pool=cls, pos=True, in=raw, snr=0.05, ord=False, noise=False, mask=0.01, rank=, den=0.0, lvl=0.0, lr=6.25e-05, drop=0.1 |
| 2 | `e311f_lite_e310_morph` | `e311f_lite_e310_morph_hc2_m010_lr625_s1` | 0.9505 | acc=0.9505, rec=0.9278/0.9496/0.9741, seed=1, pool=cls, pos=True, in=raw, snr=0.05, ord=False, noise=False, mask=0.01, rank=, den=0.0, lvl=0.0, lr=6.25e-05, drop=0.1 |
| 3 | `e311f_lite_e310_morph` | `e311f_lite_e310_morph_hc19_mask010_lr575_s1` | 0.9491 | acc=0.9491, rec=0.9305/0.9387/0.9782, seed=1, pool=cls, pos=True, in=raw, snr=0.05, ord=False, noise=False, mask=0.01, rank=, den=0.0, lvl=0.0, lr=5.75e-05, drop=0.1 |
| 4 | `e311f_lite_e310_morph` | `e311f_lite_e310_morph_hc2_m010_lr575_s1` | 0.9491 | acc=0.9491, rec=0.9305/0.9387/0.9782, seed=1, pool=cls, pos=True, in=raw, snr=0.05, ord=False, noise=False, mask=0.01, rank=, den=0.0, lvl=0.0, lr=5.75e-05, drop=0.1 |
| 5 | `e311f_lite_e310_morph` | `e311f_lite_e310_morph_hc03_base_lr625_s1` | 0.9464 | acc=0.9464, rec=0.9292/0.9346/0.9755, seed=1, pool=cls, pos=True, in=raw, snr=0.05, ord=False, noise=False, mask=False, rank=, den=0.0, lvl=0.0, lr=6.25e-05, drop=0.1 |
| 6 | `e311f_lite_e310_morph` | `e311f_lite_e310_morph_r3_lr625_seed1` | 0.9464 | acc=0.9464, rec=0.9292/0.9346/0.9755, seed=1, pool=cls, pos=True, in=raw, snr=0.05, ord=False, noise=False, mask=False, rank=, den=0.0, lvl=0.0, lr=6.25e-05, drop=0.1 |
| 7 | `e311f_lite_e310_morph` | `e311f_lite_e310_morph_hc12_ord005_lr575_s6` | 0.9455 | acc=0.9455, rec=0.9373/0.9278/0.9714, seed=6, pool=cls, pos=True, in=raw, snr=0.05, ord=True, noise=False, mask=False, rank=, den=0.0, lvl=0.0, lr=5.75e-05, drop=0.1 |
| 8 | `e311f_lite_e310_morph` | `e311f_lite_e310_morph_hc18_mask005_lr575_s1` | 0.9455 | acc=0.9455, rec=0.9278/0.9455/0.9632, seed=1, pool=cls, pos=True, in=raw, snr=0.05, ord=False, noise=False, mask=0.005, rank=, den=0.0, lvl=0.0, lr=5.75e-05, drop=0.1 |
| 9 | `e311f_lite_e310_morph` | `e311f_lite_e310_morph_hc01_base_lr575_s1` | 0.9450 | acc=0.9450, rec=0.9278/0.9319/0.9755, seed=1, pool=cls, pos=True, in=raw, snr=0.05, ord=False, noise=False, mask=False, rank=, den=0.0, lvl=0.0, lr=5.75e-05, drop=0.1 |
| 10 | `e311f_lite_e310_morph` | `e311f_lite_e310_morph_hc04_base_lr625_s6` | 0.9450 | acc=0.9450, rec=0.9237/0.9360/0.9755, seed=6, pool=cls, pos=True, in=raw, snr=0.05, ord=False, noise=False, mask=False, rank=, den=0.0, lvl=0.0, lr=6.25e-05, drop=0.1 |
| 11 | `e311f_lite_e310_morph` | `e311f_lite_e310_morph_hc14_noise005_lr575_s1` | 0.9450 | acc=0.9450, rec=0.9305/0.9292/0.9755, seed=1, pool=cls, pos=True, in=raw, snr=0.05, ord=False, noise=True, mask=False, rank=, den=0.0, lvl=0.0, lr=5.75e-05, drop=0.1 |
| 12 | `e311f_lite_e310_morph` | `e311f_lite_e310_morph_hc2_m010_lr625_s6` | 0.9450 | acc=0.9450, rec=0.9346/0.9360/0.9646, seed=6, pool=cls, pos=True, in=raw, snr=0.05, ord=False, noise=False, mask=0.01, rank=, den=0.0, lvl=0.0, lr=6.25e-05, drop=0.1 |
| 13 | `e311h_lite_relaxed_morph` | `e311h_lite_relaxed_morph_h03_snr010_pos` | 0.9103 | acc=0.9103, rec=0.9287/0.8711/0.9312, seed=0, pool=cls, pos=True, in=raw, snr=0.1, ord=False, noise=False, mask=False, rank=, den=0.0, lvl=0.0, lr=3e-05, drop=0.1 |
| 14 | `e311h_lite_relaxed_morph` | `e311h_lite_relaxed_morph_h01_pos` | 0.9095 | acc=0.9095, rec=0.9337/0.8536/0.9412, seed=0, pool=cls, pos=True, in=raw, snr=0.05, ord=False, noise=False, mask=False, rank=, den=0.0, lvl=0.0, lr=3e-05, drop=0.1 |
| 15 | `e311h_lite_relaxed_morph` | `e311h_lite_relaxed_morph_h00_anchor` | 0.9070 | acc=0.9070, rec=0.9124/0.8836/0.9249, seed=0, pool=cls, pos=False, in=raw, snr=0.05, ord=False, noise=False, mask=False, rank=, den=0.0, lvl=0.0, lr=3e-05, drop=0.1 |

## Focused Sweep Stability

Grouped across Round 2-4 E3.11f runs by learning rate and dropout.

| LR | Dropout | N | Mean Acc | Std | Min | Max |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5.75e-05 | 0.1 | 8 | 0.9423 | 0.0019 | 0.9391 | 0.9450 |
| 6.5e-05 | 0.1 | 4 | 0.9416 | 0.0020 | 0.9396 | 0.9450 |
| 6.25e-05 | 0.1 | 8 | 0.9414 | 0.0038 | 0.9346 | 0.9464 |
| 6.1e-05 | 0.1 | 4 | 0.9411 | 0.0022 | 0.9387 | 0.9446 |
| 5.5e-05 | 0.075 | 3 | 0.9410 | 0.0026 | 0.9387 | 0.9446 |
| 5.25e-05 | 0.1 | 4 | 0.9405 | 0.0022 | 0.9387 | 0.9441 |
| 6.4e-05 | 0.1 | 4 | 0.9404 | 0.0023 | 0.9373 | 0.9437 |
| 5.5e-05 | 0.1 | 4 | 0.9402 | 0.0036 | 0.9346 | 0.9437 |
| 6.75e-05 | 0.1 | 4 | 0.9396 | 0.0035 | 0.9337 | 0.9428 |
| 6e-05 | 0.1 | 4 | 0.9395 | 0.0031 | 0.9360 | 0.9428 |
| 5e-05 | 0.1 | 15 | 0.9375 | 0.0048 | 0.9251 | 0.9432 |
| 6e-05 | 0.075 | 3 | 0.9372 | 0.0026 | 0.9342 | 0.9405 |
| 3e-05 | 0.1 | 2 | 0.9357 | 0.0007 | 0.9351 | 0.9364 |

## Stage Decisions

- Keep `e311f_lite_e310_morph` as the main E3.11 data line: it is the only branch close to the historical E3.10 visual result.
- Prune `e311h_lite_relaxed_morph`: the relaxed-morph screen stayed around `0.90-0.91`, far below E3.11f.
- Prune `e311i_wide_relaxed_morph`: it has higher SQI baselines and early validation was clearly worse than E3.11f, so the wide-SNR branch is diagnostic only.
- Round 2 crossed the target with `e311f_lite_e310_morph_r2_lr5_seed1_pos` at `0.9432`.
- Round 3 improved the best single model to `0.9464` with `lr=6.25e-5`, while `lr=5.75e-5` was the most stable high-performing basin.
- Round 4 did not beat the Round 3 best; it confirmed `5.75e-5` as the most stable LR and `6.25e-5` as the highest single-run point.
- Keep the simple model recipe: CLS pooling, positional embedding, raw input, D1 warm-start, SNR head with `lambda_snr=0.05`, no denoise, no local head, no rank loss.
- Drop the weak branches from the next sweep: relaxed morphology, wide SNR, raw_robust input, cls_mean pooling, val-loss checkpoint selection, noise-type head, and class-weight tweaks.
- Stop broad model-grid tuning here unless the dataset changes; further gains are more likely from data/source audit or an ensemble diagnostic than from adding heads.

## Pruning Rules

- If a variant's first 3-6 runs stay below `0.90`, stop expanding it.
- If relaxed morphology beats E3.11f by `>=0.01`, expand that data branch.
- If wide SNR is high but SQI baselines are also high, keep it as visual/diagnostic rather than the main benchmark.
- If positional embedding helps the top E3.11f runs, keep it in round 2; otherwise remove it.
- If denoise-aware runs lose more than `0.01` versus their paired classifier-only run, remove denoise from the next grid.
