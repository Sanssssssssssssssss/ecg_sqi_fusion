# CleanBUT Bad-Core Targeted Synthetic Grid

This is a generator-target scan. CleanBUT-Core is used only as a target/diagnostic subset; original BUT 10s P1 remains the benchmark.

## Current Best CPU Fit

- Best variant: `n/a`
- Weighted score: `nan`
- Bad 64D distance: `nan` vs prior baseline `~0.748`
- Medium 64D distance: `nan` vs prior baseline `~0.311`

## Figures

- `figures/top_rules_64d_overlay.png`: CleanBUT-Core background with top targeted PTB rules.
- `figures/best_rule_64d_overlay.png`: best current candidate in the same PCA space.
- `figures/bad_core_centroid_shift.png`: class centroid gaps in CleanBUT 64D PCA.
- `figures/classwise_distance_bars.png`: class-wise distance leaderboard.

## Top No-Training Candidates

_No rows._

## Training Results

| mode | variant_id | acc | macro_f1 | good_recall | medium_recall | bad_recall | balanced_macro | clean_diag_macro | score | bad_64d_KS | medium_64d_KS | good_64d_KS | domain_separability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| quick | nl_n4200_bridge_scan_109_sc_overlap_narrow_oscillatory_co_c3d1ca649eaf | 0.6960 | 0.4904 | 0.4739 | 0.9385 | 0.0511 | 0.4149 | 0.9465 | 0.2343 | 0.5519 | 0.2473 | 0.3314 | 1.0000 |
| quick | nl_n4200_bridge_scan_109_sc_overlap_narrow_oscillatory_co_94ed3c65af32 | 0.6916 | 0.4666 | 0.5852 | 0.8434 | 0.0000 | 0.3911 | 0.9886 | 0.2343 | 0.5519 | 0.2473 | 0.3314 | 1.0000 |
| quick | nl_n4200_bridge_scan_109_sc_overlap_narrow_oscillatory_co_eeb18c048c4b | 0.6268 | 0.4260 | 0.6063 | 0.7018 | 0.0000 | 0.3606 | 0.9780 | 0.2343 | 0.5519 | 0.2473 | 0.3314 | 1.0000 |
| quick | nl_n4200_bridge_scan_099_sc_overlap_1530_hfedge_spike_cor_99f16ef78986 | 0.6296 | 0.5259 | 0.4887 | 0.7716 | 0.3479 | 0.5372 | 0.9478 | 0.2361 | 0.5964 | 0.2507 | 0.3184 | 1.0000 |
| quick | nl_n4200_bridge_scan_099_sc_overlap_1530_hfedge_spike_cor_5b9c27371d75 | 0.6444 | 0.5442 | 0.5187 | 0.7691 | 0.4161 | 0.5696 | 0.8895 | 0.2361 | 0.5964 | 0.2507 | 0.3184 | 1.0000 |
| quick | nl_n4200_bridge_scan_099_sc_overlap_1530_hfedge_spike_cor_933af4a02039 | 0.5855 | 0.4878 | 0.5360 | 0.6403 | 0.4331 | 0.5561 | 0.9721 | 0.2361 | 0.5964 | 0.2507 | 0.3184 | 1.0000 |
| quick | nl_n4200_bridge_scan_029_sc_overlap_narrow_oscillatory_co_285a9cc88e32 | 0.6588 | 0.4353 | 0.4659 | 0.8787 | 0.0000 | 0.3691 | 0.9719 | 0.2395 | 0.5605 | 0.2496 | 0.3677 | 1.0000 |
| quick | nl_n4200_bridge_scan_029_sc_overlap_narrow_oscillatory_co_f089d281670c | 0.6935 | 0.4698 | 0.6231 | 0.8159 | 0.0000 | 0.3936 | 0.9655 | 0.2395 | 0.5605 | 0.2496 | 0.3677 | 1.0000 |
| quick | nl_n4200_bridge_scan_029_sc_overlap_narrow_oscillatory_co_633b407cd67f | 0.6597 | 0.4456 | 0.5835 | 0.7836 | 0.0000 | 0.3757 | 0.9699 | 0.2395 | 0.5605 | 0.2496 | 0.3677 | 1.0000 |
| quick | nl_n4200_bridge_scan_024_sc_overlap_1530_spike_core_023_d_010974024ebf | 0.7117 | 0.4780 | 0.5712 | 0.8934 | 0.0000 | 0.4011 | 0.9477 | 0.2403 | 0.5577 | 0.2485 | 0.3184 | 1.0000 |
| quick | nl_n4200_bridge_scan_024_sc_overlap_1530_spike_core_023_d_fb8d258aebb0 | 0.7070 | 0.4774 | 0.6091 | 0.8531 | 0.0000 | 0.3971 | 0.8862 | 0.2403 | 0.5577 | 0.2485 | 0.3184 | 1.0000 |
| quick | nl_n4200_bridge_scan_024_sc_overlap_1530_spike_core_023_d_75b8be461c02 | 0.6826 | 0.4600 | 0.5549 | 0.8507 | 0.0024 | 0.3928 | 0.9565 | 0.2403 | 0.5577 | 0.2485 | 0.3184 | 1.0000 |
| quick | nl_n4200_bridge_scan_039_sc_overlap_bandlimited_disagree__85fc02f017ca | 0.6726 | 0.4593 | 0.7387 | 0.6808 | 0.0000 | 0.3846 | 0.9672 | 0.2405 | 0.6464 | 0.2496 | 0.3184 | 1.0000 |
| quick | nl_n4200_bridge_scan_039_sc_overlap_bandlimited_disagree__bd93beb00924 | 0.6301 | 0.4191 | 0.4830 | 0.8095 | 0.0000 | 0.3516 | 0.9564 | 0.2405 | 0.6464 | 0.2496 | 0.3184 | 1.0000 |
| quick | nl_n4200_bridge_scan_039_sc_overlap_bandlimited_disagree__b32b5f2e1513 | 0.7069 | 0.4751 | 0.5772 | 0.8791 | 0.0000 | 0.3917 | 0.8860 | 0.2405 | 0.6464 | 0.2496 | 0.3184 | 1.0000 |
| quick | nl_n4200_bridge_scan_094_sc_overlap_compact_pca_core_093__2d25051030f7 | 0.6833 | 0.6090 | 0.5681 | 0.8127 | 0.3090 | 0.5600 | 0.9850 | 0.2408 | 0.5882 | 0.2510 | 0.3314 | 1.0000 |
| quick | nl_n4200_bridge_scan_094_sc_overlap_compact_pca_core_093__5b7ddad8f53b | 0.6624 | 0.4449 | 0.5437 | 0.8215 | 0.0000 | 0.3796 | 0.9766 | 0.2408 | 0.5882 | 0.2510 | 0.3314 | 1.0000 |
| quick | nl_n4200_bridge_scan_094_sc_overlap_compact_pca_core_093__df89413b1a24 | 0.6531 | 0.4226 | 0.3967 | 0.9245 | 0.0000 | 0.3540 | 0.9239 | 0.2408 | 0.5882 | 0.2510 | 0.3314 | 1.0000 |
| quick | nl_n4200_bridge_scan_093_sc_overlap_compact_pca_core_092__8a3deeb9de4a | 0.6203 | 0.3893 | 0.3165 | 0.9277 | 0.0000 | 0.3167 | 0.8392 | 0.2462 | 0.6305 | 0.2473 | 0.3436 | 1.0000 |
| quick | nl_n4200_bridge_scan_093_sc_overlap_compact_pca_core_092__66cd6fd9b45a | 0.6440 | 0.4309 | 0.3967 | 0.9051 | 0.0219 | 0.3654 | 0.8665 | 0.2462 | 0.6305 | 0.2473 | 0.3436 | 1.0000 |

## Notes

- Selection uses CleanBUT train-target core features and does not inspect BUT test predictions.
- Synthetic `sqi_iSQI` remains a single-lead detector-agreement proxy.
- `all` defaults to CPU distribution fitting only; pass `--run_training` for quick/full training after visual review.
