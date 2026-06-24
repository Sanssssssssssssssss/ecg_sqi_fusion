# Waveform-only interpretable evidence - 2026-06-23

## One-line conclusion

当前最稳的论证不再是 PTB->BUT cross transfer，而是：

1. 用可解释的 PTB synthetic 规则生成数据；
2. 用 waveform-only Event-Factorized SQI Conformer 从零训练；
3. 证明 PTB clean protocol 和 BUT clean protocol 在各自域内都可以被 raw-wave 模型学到；
4. 用 feature recovery 证明模型确实学到了一批 ECG quality 因子，而不是只靠 MLP/tree/route。

Cross-dataset 结果仍然保留为 domain shift / morphology mismatch 诊断，不作为当前主结果。

## Scope and leakage rules

- Formal inference input: waveform-derived channels only.
- SQI/PCA/factor columns: generation targets, training teacher targets, and diagnostics only.
- No route/rule artifact, no feature-only MLP/tree classifier as the formal model.
- Latest self-test used random initialization; checkpoints were saved outputs only, not warm-start inputs.
- PTB-only self-test and BUT-only self-test are separate; no joint training is used in the self-test report.

## Synthetic data logic

The current main synthetic protocol is `v27_pca_subtype`.

Protocol path:

`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v27_pca_subtype_gm_featurematched\protocol_v27_pca_subtype_pc3000_s20260622`

Core metadata:

- Base protocol: `event_xds_aligned_v20_bad_subtype_featurematched\protocol_v20_pc3000_s20260621`
- BUT reference policy: `clean_but_protocols\margin_ge_5s_drop_outlier`
- Target scope: BUT train+val good/medium feature PCA subtypes
- BUT waveform copy: no
- BUT test use in generation: no
- PCA explained ratio: PC1 `0.37796`, PC2 `0.20188`, PC3 `0.13553`

Good/medium target subtypes:

| class | subtype | target share | interpretation |
| --- | --- | ---: | --- |
| good | good_low_pc1_shell | 0.107 | outer good shell |
| good | good_core_1 | 0.315 | stable good body |
| good | good_core_2 | 0.403 | stable good body |
| good | good_pc3_morph_tail | 0.176 | morphology-tail good |
| medium | medium_goodlike_low_pc1 | 0.056 | medium near good-like shell |
| medium | medium_goodlike_low_pc1 | 0.119 | medium near good-like shell |
| medium | medium_high_pc2_detail | 0.252 | detail/noise medium |
| medium | medium_high_pc2_detail | 0.157 | detail/noise medium |
| medium | medium_high_pc2_detail | 0.415 | main high-PC2 medium shell |

PTB synthetic split counts:

| split | good | medium | bad |
| --- | ---: | ---: | ---: |
| train | 2105 | 2014 | 2100 |
| val | 443 | 480 | 450 |
| test | 452 | 506 | 450 |

Interpretation:

- The synthetic data is not "random SNR only" anymore.
- Good/medium are generated to cover interpretable PCA/SQI shells derived from BUT train+val.
- Bad uses the earlier v20 subtype-matched bad protocol as the guardrail/body.
- PCA is used as a generation/audit coordinate, not as a formal inference feature.

## Distribution closure result

Report:

`E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\gm_distribution_closure_20260622.md`

Main result:

| protocol | interpretation | PTB->BUT acc | good recall | medium recall | bad recall |
| --- | --- | ---: | ---: | ---: | ---: |
| v27_subtype E1 | best formal cross candidate | 0.743357 | 0.988048 | 0.610496 | 1.000000 |
| v28_anchor E1 | visually smoother PCA, worse transfer | 0.690216 | 0.993028 | 0.526240 | 1.000000 |
| v33_naturalmedium E1 | more natural medium, still lower | 0.734605 | 0.988048 | 0.597015 | 1.000000 |
| v35_oracle_allbut E1 | diagnostic leakage; uses BUT test feature anchors | 0.685839 | 0.989044 | 0.521425 | 1.000000 |
| v36_naturalheavy E1 | natural-heavy quota, transfer collapse | 0.599875 | 0.988048 | 0.389504 | 1.000000 |

Important interpretation:

- Visually closer PCA is necessary but not sufficient.
- Even oracle access to BUT test feature anchors did not solve transfer, so the remaining issue is not only feature distribution matching.
- The current cross failure points to synthetic waveform morphology/source mismatch, especially 111001-style medium windows.
- Therefore, cross should be treated as a diagnostic axis while the current method claim focuses on interpretable generation and waveform model learnability.

Key distribution figures:

![Good/medium PCA distribution panels](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/gm_pca_distribution_fit_diagnostics/gm_pca_distribution_panels.png)

![BUT 111001 medium vs nearest synthetic waveforms](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/gm_pca_distribution_fit_diagnostics/gm_111001_medium_pca_nearest_waveforms.png)

![Nearest-neighbor feature deltas](E:/GPTProject2/ecg/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/gm_pca_distribution_fit_diagnostics/gm_111001_medium_pca_nearest_feature_deltas.png)

## Model structure

Implementation:

`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\run_event_factorized_sqi_conformer.py`

Model:

`EventFactorizedSQIConformer`

This is not UFormer. It is an external-only Event-Factorized SQI Conformer.

Input:

- 8-channel waveform-derived view from the clean protocol loader.
- No tabular SQI/PCA/factor inputs at inference.

Token path:

- High-resolution stem:
  - Conv1d kernel 11, stride 2
  - GroupNorm + GELU
  - Conv1d kernel 7, stride 1
  - GroupNorm + GELU
- Context downsample:
  - Conv1d kernel 9, stride 4
  - GroupNorm + GELU
- Positional encoding is used before Conformer blocks.
- Conformer blocks combine self-attention and depthwise temporal convolution.

Query tokens:

- `QRS`
- `RR_TEMPLATE`
- `BASELINE`
- `CONTACT_RESET`
- `DETAIL_NOISE`
- `GLOBAL_MORPH`
- `GM_BOUNDARY`
- `BAD_STRESS`

Heads:

- Factor head reads the first six mechanism tokens.
- GM head reads `GM_BOUNDARY`.
- Bad head reads `BAD_STRESS`.
- Artifact/severity heads are present for variants that enable artifact auxiliary loss.
- Detector agreement in the reported factor vector is the same soft event-agreement statistic derived from local QRS maps, avoiding the earlier "reported but unsupervised factor slot" issue.

Classification probability:

```text
P(bad) = b
P(medium) = (1 - b) * m
P(good) = (1 - b) * (1 - m)
```

The selected self-test candidate is:

`E1_query_only`

It enables query tokens and hierarchical classification, but disables high-res cross-attention, local-map supervision, and artifact auxiliary loss. This is intentionally clean: it tests whether factorized query readout alone can learn the clean protocols from waveform-derived channels.

## Single-domain self-test

Report:

`E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_sqi_conformer\event_factorized_single_domain_selftest_v27_interpretability_e1_report.md`

Outputs:

`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v27_interpretability_e1`

Results:

| domain | bucket | acc | macro-F1 | good recall | medium recall | bad recall | record-macro supported F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PTB-only | test | 0.972301 | 0.972348 | 1.000000 | 1.000000 | 0.913333 | 0.999493 |
| PTB-only | all | 0.970889 | 0.970799 | 1.000000 | 0.999667 | 0.913000 | 0.999735 |
| BUT-only | test | 0.966865 | 0.974546 | 0.999004 | 0.949446 | 1.000000 | 0.961052 |
| BUT-only | all | 0.979838 | 0.981730 | 0.995102 | 0.939505 | 0.999755 | 0.971501 |

Caveat:

- The BUT validation split has only one bad sample, so val bad statistics are not meaningful.
- BUT test has 118 bad, 1004 good, and 2077 medium rows, so test/all are the more useful summaries here.

Interpretation:

- PTB synthetic clean protocol is learnable by the waveform model.
- BUT clean protocol is also learnable by the same waveform model family.
- This supports the current methodological claim: ECG quality classes are separable from waveform-derived input under a clean, interpretable protocol.

## Feature recovery

Feature recovery file:

`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v27_interpretability_e1\single_domain_selftest_feature_recovery.csv`

Selected test correlations:

| feature | PTB test corr_all | BUT test corr_all | current interpretation |
| --- | ---: | ---: | --- |
| baseline_step | 0.984323 | 0.734949 | learned well; usable interpretability evidence |
| sqi_basSQI | 0.984138 | 0.786228 | learned well overall; bad-class BUT still weak |
| non_qrs_diff_p95 | 0.744252 | 0.897228 | learned well; detail/noise evidence is strong |
| qrs_band_ratio | 0.000000 | 0.574541 | not reliable in PTB protocol; moderate in BUT |
| flatline_ratio | 0.163125 | 0.906165 | weak in PTB global recovery, strong in BUT |
| amplitude_entropy | 0.294579 | 0.922443 | weak in PTB, strong in BUT |
| template_corr | 0.938408 | 0.454866 | strong in PTB, partial in BUT |
| qrs_visibility | 0.364270 | 0.546571 | partially learned, still hard |
| detector_agreement | -0.186411 | 0.266470 | not stable enough; remains a key weak factor |

Strong evidence:

- `baseline_step`, `sqi_basSQI`, and `non_qrs_diff_p95` are learnable from waveform-derived input.
- BUT-only also shows strong recovery for `flatline_ratio` and `amplitude_entropy`.
- These support the argument that the model is learning interpretable quality evidence, not just fitting class labels.

Remaining weak points:

- `detector_agreement` is still unstable.
- `qrs_visibility` is only partially recovered.
- Per-class bad recovery is weaker than all-class recovery for several factors.
- This means the next model improvement should focus on event/QRS supervision and natural bad morphology, not on adding a feature MLP.

## Methodological interpretation

What we can claim now:

- The synthetic generator is interpretable: it is built from subtype quotas and waveform-computable SQI/PCA shells, not arbitrary black-box augmentation.
- The Event-Factorized SQI Conformer is waveform-only at inference.
- The model can learn both PTB and BUT clean protocols from scratch with high accuracy.
- The model recovers several meaningful ECG quality factors from raw waveform-derived channels.

What we should not claim:

- We should not claim PTB->BUT is solved; current best formal cross result is still about 0.74.
- We should not claim PCA/atlas/KNN geometry is a formal model input.
- We should not use rule artifacts or feature-only MLP/tree as final model claims.
- We should not treat visually smooth PCA fit as proof of transfer, because v28/v35 showed that it can look better and transfer worse.

## Recommended next direction

The next productive direction is source-morphology-aware generation plus stronger event supervision:

1. Select PTB clean source windows whose uncorrupted morphology is already close to BUT natural medium/bad examples.
2. Generate artifacts on those sources without spike-train or burst shortcuts.
3. Improve QRS/RR/event supervision so `qrs_visibility` and `detector_agreement` become stable.
4. Keep self-test and feature recovery as the primary evidence package.
5. Keep PTB->BUT cross as a domain-shift stress test, not as the main method claim.

## Key files for review

Runner:

`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\run_event_factorized_sqi_conformer.py`

Single-domain report:

`E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_sqi_conformer\event_factorized_single_domain_selftest_v27_interpretability_e1_report.md`

Single-domain outputs:

`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_factorized_single_domain_selftest_v27_interpretability_e1`

Distribution closure report:

`E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\gm_distribution_closure_20260622.md`

Distribution visuals:

`E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\gm_pca_distribution_fit_diagnostics`

v27 synthetic protocol:

`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\event_xds_aligned_v27_pca_subtype_gm_featurematched\protocol_v27_pca_subtype_pc3000_s20260622`
