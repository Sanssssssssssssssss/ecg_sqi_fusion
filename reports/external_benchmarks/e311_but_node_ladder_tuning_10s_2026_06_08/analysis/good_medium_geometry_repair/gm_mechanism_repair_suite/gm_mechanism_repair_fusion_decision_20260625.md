# GM Mechanism Repair Fusion Decision

Date: 2026-06-25

Policy: `ptb_v112_gm_buffered_large_hybrid_s20260741`

## Decision

Mainline candidate: `E6_factor_fused_gm`

Why: it remains the best clean-test checkpoint family after the fusion sweep. It has the highest mean accuracy and macro-F1, and the lowest total good/medium mutual errors.

Balanced ablation to preserve: `E14_e6_medguard_lowgain`

Why: it does not beat E6 on accuracy, but it slightly improves good/medium balanced recall. Keep it as an ablation for “medium guard + reduced factor injection,” not as the mainline.

## E4-E9 Baseline Sweep

| candidate | acc | macro-F1 | good | medium | bad | GM balanced | GM errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E6_factor_fused_gm | 0.9220 | 0.9056 | 0.8947 | 0.8537 | 0.9723 | 0.8742 | 596 |
| E5_factor_contract_only | 0.9199 | 0.9025 | 0.8635 | 0.8640 | 0.9763 | 0.8637 | 616 |
| E4_v112_lowaux_lr15e4 | 0.9189 | 0.9011 | 0.8592 | 0.8641 | 0.9764 | 0.8617 | 633 |
| E8_pairrank_hardsampler | 0.9160 | 0.8973 | 0.8783 | 0.8376 | 0.9772 | 0.8579 | 665 |
| E7_family_moe_condsubtype | 0.9133 | 0.8943 | 0.8569 | 0.8514 | 0.9733 | 0.8542 | 687 |
| E9_beat_background_tokens | 0.8638 | 0.8319 | 0.8786 | 0.6919 | 0.9564 | 0.7853 | 1212 |

## E10-E14 Fusion Sweep

| candidate | acc | macro-F1 | good | medium | bad | GM balanced | GM errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E14_e6_medguard_lowgain | 0.9214 | 0.9049 | 0.8904 | 0.8601 | 0.9702 | 0.8753 | 613 |
| E13_e6_boundary_aux_stronger | 0.9197 | 0.9021 | 0.8475 | 0.8788 | 0.9747 | 0.8632 | 624 |
| E12_e6_pairrank_only | 0.9195 | 0.9024 | 0.8881 | 0.8529 | 0.9711 | 0.8705 | 625 |
| E10_e6_medguard | 0.9172 | 0.8991 | 0.8663 | 0.8575 | 0.9724 | 0.8619 | 634 |
| E11_e6_lowgain | 0.9155 | 0.8974 | 0.8588 | 0.8482 | 0.9791 | 0.8535 | 641 |

## Mechanism Readout

Useful mechanisms:

- `E5` factor contract fix is mandatory. It corrected the train/decode/eval mismatch for waveform-computable factors such as `qrs_visibility`, `qrs_band_ratio`, and `sqi_basSQI`.
- `E6` factor-fused GM head is useful. It improves total accuracy and reduces good/medium mutual errors compared with E4/E5.
- `E14` is a mild balancing ablation. It slightly improves GM balanced recall, but loses total accuracy and increases GM mutual errors, so it should not replace E6.

Mechanisms not promoted:

- Medium class-weight guard alone (`E10`) makes good/medium recall oscillate by fold and epoch.
- Lowering factor injection (`E11`) reduces some over-correction but underperforms E6.
- Pairrank alone (`E12`) is mildly useful but not enough without a true pair-aware batch sampler.
- Stronger subtype auxiliary loss (`E13`) creates local wins but does not improve the global boundary.
- Family MoE/conditional subtype (`E7/E8`) and beat/background tokens (`E9`) are not ready for mainline.

Current bottleneck:

- Decoded factor recovery is already strong for the important waveform-computable SQI factors, usually around 0.95 correlation for `qrs_visibility`, `qrs_band_ratio`, `sqi_basSQI`, `template_corr`, and `non_qrs_rms_ratio`.
- Local event supervision is still weak: QRS event Dice is only about 0.34-0.36 and peak-count error remains around 80-95. Beat/local mechanisms should not be trusted until the event target and event head are fixed.
- Boundary-family supervision is also weak, with boundary balanced accuracy only around 0.55-0.57 in the fusion sweep. This explains why adding more subtype/family loss did not translate into a clean accuracy jump.

## Output Locations

Baseline E4-E9 metrics backup:

`outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/gm_mechanism_repair_suite/candidate_metrics_e4e9_baseline_20260625.csv`

Fusion E10-E14 metrics:

`outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/gm_mechanism_repair_suite/candidate_metrics.csv`

Generated report:

`reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/gm_mechanism_repair_suite/gm_mechanism_repair_suite_report.md`

Waveform panels:

`reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/gm_mechanism_repair_suite/*_waveform_panel.png`

Runner:

`outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/run_gm_mechanism_repair_suite.py`

## Next Experimental Direction

Do not keep sweeping class weights or adding broad subtype loss. The most promising next step is still E6 as the mainline, with one of these targeted repairs:

1. Implement a true pair-aware batch sampler for good/medium sibling examples, then retest pairrank on top of E6.
2. Fix local QRS/event targets before using beat/background tokens again.
3. Rebuild the boundary-family target so it predicts a reliable waveform mechanism; the current family head is too noisy to guide classification.
