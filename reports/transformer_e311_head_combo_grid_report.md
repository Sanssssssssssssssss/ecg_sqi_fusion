# E3.11 Head Combination Grid

This grid searches task-head combinations on the E3.11f mainline data, without changing the backbone or labels.
The fixed strong recipe is CLS pooling, positional embedding, raw input, D1 warm-start, and validation-accuracy checkpoint selection.

Rationale references used for the grid design:

- MTECG: ECG segment tokens with learnable positional embeddings and masked-autoencoder style representation learning: https://arxiv.org/abs/2309.07136
- UniTS: time-series models can share parameters across classification, imputation, and related tasks: https://github.com/mims-harvard/UniTS
- SwinDAE: ECG quality assessment can benefit from pairing Transformer features with denoising-autoencoder objectives: https://pubmed.ncbi.nlm.nih.gov/37698969/

Historical anchors:

- E3.10 best single visual: `0.9402`
- E3.11f best before head-combo grid: `0.9464` (`r3_lr625_seed1`)
- E3.11f stable basin: `lr=5.75e-5`, dropout `0.10`, mean about `0.9423` across prior seeds

## Top Runs

| Rank | Run | Family | Test Acc | Recall G/M/B | Head/Loss Summary |
| ---: | --- | --- | ---: | --- | --- |
| 1 | `e311f_lite_e310_morph_hc22_mask010_lr625_s1` | mask_aux | 0.9505 | 0.9278/0.9496/0.9741 | seed=1, lr=6.25e-05, drop=0.1, snr=0.05, ord=False, noise=False, mask=0.01, rank=, den=0.0, lvl=0.0, uncert=0, gw/mw/bw=1.0/1.0/1.0, ls=0.0, wd=0.03 |
| 2 | `e311f_lite_e310_morph_hc2_m010_lr625_s1` | mask_aux | 0.9505 | 0.9278/0.9496/0.9741 | seed=1, lr=6.25e-05, drop=0.1, snr=0.05, ord=False, noise=False, mask=0.01, rank=, den=0.0, lvl=0.0, uncert=0, gw/mw/bw=1.0/1.0/1.0, ls=0.0, wd=0.03 |
| 3 | `e311f_lite_e310_morph_hc19_mask010_lr575_s1` | mask_aux | 0.9491 | 0.9305/0.9387/0.9782 | seed=1, lr=5.75e-05, drop=0.1, snr=0.05, ord=False, noise=False, mask=0.01, rank=, den=0.0, lvl=0.0, uncert=0, gw/mw/bw=1.0/1.0/1.0, ls=0.0, wd=0.03 |
| 4 | `e311f_lite_e310_morph_hc2_m010_lr575_s1` | mask_aux | 0.9491 | 0.9305/0.9387/0.9782 | seed=1, lr=5.75e-05, drop=0.1, snr=0.05, ord=False, noise=False, mask=0.01, rank=, den=0.0, lvl=0.0, uncert=0, gw/mw/bw=1.0/1.0/1.0, ls=0.0, wd=0.03 |
| 5 | `e311f_lite_e310_morph_hc03_base_lr625_s1` | snr_baseline | 0.9464 | 0.9292/0.9346/0.9755 | seed=1, lr=6.25e-05, drop=0.1, snr=0.05, ord=False, noise=False, mask=False, rank=, den=0.0, lvl=0.0, uncert=0, gw/mw/bw=1.0/1.0/1.0, ls=0.0, wd=0.03 |
| 6 | `e311f_lite_e310_morph_hc12_ord005_lr575_s6` | ord_aux | 0.9455 | 0.9373/0.9278/0.9714 | seed=6, lr=5.75e-05, drop=0.1, snr=0.05, ord=0.05, noise=False, mask=False, rank=, den=0.0, lvl=0.0, uncert=0, gw/mw/bw=1.0/1.0/1.0, ls=0.0, wd=0.03 |
| 7 | `e311f_lite_e310_morph_hc18_mask005_lr575_s1` | mask_aux | 0.9455 | 0.9278/0.9455/0.9632 | seed=1, lr=5.75e-05, drop=0.1, snr=0.05, ord=False, noise=False, mask=0.005, rank=, den=0.0, lvl=0.0, uncert=0, gw/mw/bw=1.0/1.0/1.0, ls=0.0, wd=0.03 |
| 8 | `e311f_lite_e310_morph_hc01_base_lr575_s1` | snr_baseline | 0.9450 | 0.9278/0.9319/0.9755 | seed=1, lr=5.75e-05, drop=0.1, snr=0.05, ord=False, noise=False, mask=False, rank=, den=0.0, lvl=0.0, uncert=0, gw/mw/bw=1.0/1.0/1.0, ls=0.0, wd=0.03 |
| 9 | `e311f_lite_e310_morph_hc04_base_lr625_s6` | snr_baseline | 0.9450 | 0.9237/0.9360/0.9755 | seed=6, lr=6.25e-05, drop=0.1, snr=0.05, ord=False, noise=False, mask=False, rank=, den=0.0, lvl=0.0, uncert=0, gw/mw/bw=1.0/1.0/1.0, ls=0.0, wd=0.03 |
| 10 | `e311f_lite_e310_morph_hc14_noise005_lr575_s1` | noise_aux | 0.9450 | 0.9305/0.9292/0.9755 | seed=1, lr=5.75e-05, drop=0.1, snr=0.05, ord=False, noise=0.05, mask=False, rank=, den=0.0, lvl=0.0, uncert=0, gw/mw/bw=1.0/1.0/1.0, ls=0.0, wd=0.03 |
| 11 | `e311f_lite_e310_morph_hc2_m010_lr625_s6` | mask_aux | 0.9450 | 0.9346/0.9360/0.9646 | seed=6, lr=6.25e-05, drop=0.1, snr=0.05, ord=False, noise=False, mask=0.01, rank=, den=0.0, lvl=0.0, uncert=0, gw/mw/bw=1.0/1.0/1.0, ls=0.0, wd=0.03 |
| 12 | `e311f_lite_e310_morph_hc20_mask020_lr575_s1` | mask_aux | 0.9441 | 0.9237/0.9292/0.9796 | seed=1, lr=5.75e-05, drop=0.1, snr=0.05, ord=False, noise=False, mask=0.02, rank=, den=0.0, lvl=0.0, uncert=0, gw/mw/bw=1.0/1.0/1.0, ls=0.0, wd=0.03 |
| 13 | `e311f_lite_e310_morph_hc2_m010_lr61_s0` | mask_aux | 0.9441 | 0.9360/0.9264/0.9700 | seed=0, lr=6.1e-05, drop=0.1, snr=0.05, ord=False, noise=False, mask=0.01, rank=, den=0.0, lvl=0.0, uncert=0, gw/mw/bw=1.0/1.0/1.0, ls=0.0, wd=0.03 |
| 14 | `e311f_lite_e310_morph_hc2_m010_lr61_s1` | mask_aux | 0.9441 | 0.9251/0.9441/0.9632 | seed=1, lr=6.1e-05, drop=0.1, snr=0.05, ord=False, noise=False, mask=0.01, rank=, den=0.0, lvl=0.0, uncert=0, gw/mw/bw=1.0/1.0/1.0, ls=0.0, wd=0.03 |
| 15 | `e311f_lite_e310_morph_hc02_base_lr575_s6` | snr_baseline | 0.9437 | 0.9332/0.9264/0.9714 | seed=6, lr=5.75e-05, drop=0.1, snr=0.05, ord=False, noise=False, mask=False, rank=, den=0.0, lvl=0.0, uncert=0, gw/mw/bw=1.0/1.0/1.0, ls=0.0, wd=0.03 |
| 16 | `e311f_lite_e310_morph_hc16_noise002_lr625_s1` | noise_aux | 0.9437 | 0.9360/0.9210/0.9741 | seed=1, lr=6.25e-05, drop=0.1, snr=0.05, ord=False, noise=0.02, mask=False, rank=, den=0.0, lvl=0.0, uncert=0, gw/mw/bw=1.0/1.0/1.0, ls=0.0, wd=0.03 |
| 17 | `e311f_lite_e310_morph_hc2_m010_lr575_s2` | mask_aux | 0.9437 | 0.9305/0.9264/0.9741 | seed=2, lr=5.75e-05, drop=0.1, snr=0.05, ord=False, noise=False, mask=0.01, rank=, den=0.0, lvl=0.0, uncert=0, gw/mw/bw=1.0/1.0/1.0, ls=0.0, wd=0.03 |
| 18 | `e311f_lite_e310_morph_hc07_snr010_lr575_s1` | snr_lambda | 0.9432 | 0.9142/0.9401/0.9755 | seed=1, lr=5.75e-05, drop=0.1, snr=0.1, ord=False, noise=False, mask=False, rank=, den=0.0, lvl=0.0, uncert=0, gw/mw/bw=1.0/1.0/1.0, ls=0.0, wd=0.03 |
| 19 | `e311f_lite_e310_morph_hc21_mask050_lr575_s1` | mask_aux | 0.9432 | 0.9155/0.9428/0.9714 | seed=1, lr=5.75e-05, drop=0.1, snr=0.05, ord=False, noise=False, mask=0.05, rank=, den=0.0, lvl=0.0, uncert=0, gw/mw/bw=1.0/1.0/1.0, ls=0.0, wd=0.03 |
| 20 | `e311f_lite_e310_morph_hc13_noise002_lr575_s1` | noise_aux | 0.9428 | 0.9332/0.9169/0.9782 | seed=1, lr=5.75e-05, drop=0.1, snr=0.05, ord=False, noise=0.02, mask=False, rank=, den=0.0, lvl=0.0, uncert=0, gw/mw/bw=1.0/1.0/1.0, ls=0.0, wd=0.03 |
| 21 | `e311f_lite_e310_morph_hc2_m010_lr625_s2` | mask_aux | 0.9423 | 0.9278/0.9346/0.9646 | seed=2, lr=6.25e-05, drop=0.1, snr=0.05, ord=False, noise=False, mask=0.01, rank=, den=0.0, lvl=0.0, uncert=0, gw/mw/bw=1.0/1.0/1.0, ls=0.0, wd=0.03 |
| 22 | `e311f_lite_e310_morph_hc2_m010_lr625_s5` | mask_aux | 0.9423 | 0.9278/0.9305/0.9687 | seed=5, lr=6.25e-05, drop=0.1, snr=0.05, ord=False, noise=False, mask=0.01, rank=, den=0.0, lvl=0.0, uncert=0, gw/mw/bw=1.0/1.0/1.0, ls=0.0, wd=0.03 |
| 23 | `e311f_lite_e310_morph_hc2_m010_lr575_s0` | mask_aux | 0.9414 | 0.9251/0.9264/0.9728 | seed=0, lr=5.75e-05, drop=0.1, snr=0.05, ord=False, noise=False, mask=0.01, rank=, den=0.0, lvl=0.0, uncert=0, gw/mw/bw=1.0/1.0/1.0, ls=0.0, wd=0.03 |
| 24 | `e311f_lite_e310_morph_hc2_m010_lr575_s3` | mask_aux | 0.9414 | 0.9373/0.9264/0.9605 | seed=3, lr=5.75e-05, drop=0.1, snr=0.05, ord=False, noise=False, mask=0.01, rank=, den=0.0, lvl=0.0, uncert=0, gw/mw/bw=1.0/1.0/1.0, ls=0.0, wd=0.03 |
| 25 | `e311f_lite_e310_morph_hc05_snr003_lr575_s1` | snr_lambda | 0.9410 | 0.9087/0.9428/0.9714 | seed=1, lr=5.75e-05, drop=0.1, snr=0.03, ord=False, noise=False, mask=False, rank=, den=0.0, lvl=0.0, uncert=0, gw/mw/bw=1.0/1.0/1.0, ls=0.0, wd=0.03 |
| 26 | `e311f_lite_e310_morph_hc10_ord010_lr575_s1` | ord_aux | 0.9410 | 0.9305/0.9142/0.9782 | seed=1, lr=5.75e-05, drop=0.1, snr=0.05, ord=0.1, noise=False, mask=False, rank=, den=0.0, lvl=0.0, uncert=0, gw/mw/bw=1.0/1.0/1.0, ls=0.0, wd=0.03 |
| 27 | `e311f_lite_e310_morph_hc09_ord005_lr575_s1` | ord_aux | 0.9405 | 0.9346/0.9101/0.9768 | seed=1, lr=5.75e-05, drop=0.1, snr=0.05, ord=0.05, noise=False, mask=False, rank=, den=0.0, lvl=0.0, uncert=0, gw/mw/bw=1.0/1.0/1.0, ls=0.0, wd=0.03 |
| 28 | `e311f_lite_e310_morph_hc23_mask010_lr575_s6` | mask_aux | 0.9405 | 0.9346/0.9114/0.9755 | seed=6, lr=5.75e-05, drop=0.1, snr=0.05, ord=False, noise=False, mask=0.01, rank=, den=0.0, lvl=0.0, uncert=0, gw/mw/bw=1.0/1.0/1.0, ls=0.0, wd=0.03 |
| 29 | `e311f_lite_e310_morph_hc2_m010_lr61_s3` | mask_aux | 0.9405 | 0.9414/0.9142/0.9659 | seed=3, lr=6.1e-05, drop=0.1, snr=0.05, ord=False, noise=False, mask=0.01, rank=, den=0.0, lvl=0.0, uncert=0, gw/mw/bw=1.0/1.0/1.0, ls=0.0, wd=0.03 |
| 30 | `e311f_lite_e310_morph_hc17_noise002_lr575_s6` | noise_aux | 0.9401 | 0.9101/0.9401/0.9700 | seed=6, lr=5.75e-05, drop=0.1, snr=0.05, ord=False, noise=0.02, mask=False, rank=, den=0.0, lvl=0.0, uncert=0, gw/mw/bw=1.0/1.0/1.0, ls=0.0, wd=0.03 |

## Family Summary

| Family | N | Mean Acc | Std | Best Acc | Best Run |
| --- | ---: | ---: | ---: | ---: | --- |
| mask_aux | 22 | 0.9430 | 0.0041 | 0.9505 | `e311f_lite_e310_morph_hc22_mask010_lr625_s1` |
| snr_baseline | 4 | 0.9450 | 0.0010 | 0.9464 | `e311f_lite_e310_morph_hc03_base_lr625_s1` |
| ord_aux | 5 | 0.9411 | 0.0024 | 0.9455 | `e311f_lite_e310_morph_hc12_ord005_lr575_s6` |
| noise_aux | 5 | 0.9419 | 0.0026 | 0.9450 | `e311f_lite_e310_morph_hc14_noise005_lr575_s1` |
| snr_lambda | 3 | 0.9410 | 0.0019 | 0.9432 | `e311f_lite_e310_morph_hc07_snr010_lr575_s1` |
| no_snr_control | 1 | 0.9355 | 0.0000 | 0.9355 | `e311f_lite_e310_morph_hc00_no_snr_lr575_s1` |

## Pure Local-Mask Stability

| Setting | N | Mean Acc | Std | Min | Max | Best Run |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| mask=0.01, lr=6.25e-05 | 9 | 0.9427 | 0.0048 | 0.9360 | 0.9505 | `e311f_lite_e310_morph_hc22_mask010_lr625_s1` |
| mask=0.01, lr=5.75e-05 | 6 | 0.9442 | 0.0036 | 0.9405 | 0.9491 | `e311f_lite_e310_morph_hc19_mask010_lr575_s1` |
| mask=0.005, lr=5.75e-05 | 1 | 0.9455 | 0.0000 | 0.9455 | 0.9455 | `e311f_lite_e310_morph_hc18_mask005_lr575_s1` |
| mask=0.02, lr=5.75e-05 | 1 | 0.9441 | 0.0000 | 0.9441 | 0.9441 | `e311f_lite_e310_morph_hc20_mask020_lr575_s1` |
| mask=0.01, lr=6.1e-05 | 4 | 0.9411 | 0.0035 | 0.9355 | 0.9441 | `e311f_lite_e310_morph_hc2_m010_lr61_s0` |
| mask=0.05, lr=5.75e-05 | 1 | 0.9432 | 0.0000 | 0.9432 | 0.9432 | `e311f_lite_e310_morph_hc21_mask050_lr575_s1` |

## Live Decision Rules

- If SNR-only remains best, keep the model simple and spend future effort on data/source audit.
- If ordinal improves medium recall without hurting bad recall, expand ordinal around `lambda_ord=0.03-0.05`.
- If local mask helps, keep it at low weight only; the current target is injected-noise envelope, so high weights may overfit synthetic placement.
- If denoise/level auxiliary improves classification by at least `0.003`, rerun the best denoise setting across multiple seeds.
- If noise-type head is flat or negative, drop it; E3.11f uses only `em/ma/mix`, so texture supervision may be a distraction.
