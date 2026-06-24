# Good/Medium Distribution Closure Notes - 2026-06-22

## Scope

This note summarizes the latest external-only distribution fitting loop.  All
formal model inputs remain waveform-derived channels.  SQI/PCA features are used
only for synthetic generation targets, diagnostics, and visual audits.

## Main Finding

Making the PCA distribution visually closer to BUT is necessary, but not
sufficient.  The best PTB->BUT result still comes from `v27_subtype`, not from
the visually smoother or more feature-forced variants.

The failure mode is now clearer: several variants become PCA-near by creating
synthetic waveform shortcuts, especially periodic spike trains or bursty detail
patterns.  The model learns these shortcuts and transfers worse to the natural
BUT 111001 medium windows.

## Key Results

| protocol | note | PTB->BUT test acc | good recall | medium recall | bad recall |
| --- | --- | ---: | ---: | ---: | ---: |
| v27_subtype E1 | best formal so far | 0.743357 | 0.988048 | 0.610496 | 1.000000 |
| v28_anchor E1 | smoother PCA shell, worse pooled test | 0.690216 | 0.993028 | 0.526240 | 1.000000 |
| v33_naturalmedium E1 | better feature stability, still below v27 | 0.734605 | 0.988048 | 0.597015 | 1.000000 |
| v34_mixture E1 | v27+v28+v33 mixture; self-test strong, transfer worse | 0.659581 | 0.981076 | 0.484834 | 1.000000 |
| v35_oracle_allbut E1 | diagnostic leakage: uses BUT test features as anchors | 0.685839 | 0.989044 | 0.521425 | 1.000000 |
| v36_naturalheavy E1 | heavier natural-medium quota; transfer collapses | 0.599875 | 0.988048 | 0.389504 | 1.000000 |

## Diagnostic Evidence

- PCA panels and metrics:
  `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\gm_pca_distribution_fit_diagnostics\gm_pca_distribution_fit_diagnostics.md`
- BUT 111001 medium vs PCA-nearest synthetic waveforms:
  `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\gm_pca_distribution_fit_diagnostics\gm_111001_medium_pca_nearest_waveforms.png`
- Feature deltas for the same nearest-neighbor rows:
  `E:\GPTProject2\ecg\reports\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\gm_pca_distribution_fit_diagnostics\gm_111001_medium_pca_nearest_feature_deltas.png`

## Interpretation

1. `v28_anchor` proves continuous PCA coverage alone does not solve transfer.
2. `v35_oracle_allbut` proves even target-distribution access to BUT test
   features does not solve transfer; this points away from pure split shift and
   toward synthetic waveform morphology mismatch.
3. `v33/v36` prove simply increasing natural-looking medium modes is not enough.
   The synthetic source morphology still differs from natural BUT medium, and
   heavier synthetic natural modes make PTB->BUT worse.
4. `v27_subtype` remains the best formal candidate because its discrete subtype
   matching gives a more useful decision boundary, even though the scatter plot is
   less visually continuous.

## Next Direction

Stop spending cycles on single-feature forcing or mixture protocols.  The next
useful work should move one level lower:

- identify PTB source windows whose uncorrupted morphology is already closest to
  BUT 111001 medium before augmentation;
- create beat/template variability from those source windows without introducing
  synthetic spike-train shortcuts;
- audit local morphology with waveform panels before training;
- keep `v27_subtype` as the formal baseline until a new source-morphology-aware
  generator beats it on PTB->BUT test.

