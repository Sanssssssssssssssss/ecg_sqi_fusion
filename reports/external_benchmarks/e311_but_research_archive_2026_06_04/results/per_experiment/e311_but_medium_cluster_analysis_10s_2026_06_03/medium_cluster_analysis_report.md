# BUT Medium Cluster Analysis 10s

## Technical Summary

- BUT processed data has `32956` windows; this analysis uses the existing morphology feature sample with balanced class sampling from the formal 10s P1 protocol.
- Medium projects to `t=0.493` on the good-bad centroid axis, but its perpendicular ratio is `0.292`. Interpretation: `medium_has_independent_component`.
- Holdout morphology-feature classifier macro-F1 is `0.912`; medium-vs-rest F1 is `0.831`.
- The current generator direction is correct only partly: feature audits move toward BUT, but current grid rows still trade medium and bad off rather than forming the medium-specific cluster.

## Medium Is Not Just Halfway Between Good And Bad

The good-bad axis test asks whether class 2 lies on the straight line from class 1 to class 3 in robust-scaled morphology feature space. A large perpendicular component means medium has its own morphology signature.

- good-medium centroid distance: `3.205`
- medium-bad centroid distance: `3.270`
- good-bad centroid distance: `5.589`
- medium perpendicular / good-bad distance: `0.292`

## Features Where Medium Behaves Like An Independent State

| feature | good median | medium median | bad median | medium position | unique score |
|---|---:|---:|---:|---|---:|
| `local_deriv_spike_frac` | 0.0496 | 0.0600 | 0.0056 | -0.236 | 0.867 |
| `deriv_rms` | 0.3160 | 0.2569 | 0.4177 | -0.580 | 0.863 |
| `baseline_wander` | 0.3108 | 0.3464 | 0.0157 | -0.121 | 0.827 |
| `qrs_slope_proxy` | 0.6533 | 0.5356 | 0.6762 | -5.161 | 0.435 |
| `prominence_cv` | 0.7068 | 0.7087 | 0.2527 | -0.004 | 0.022 |

## Strongest Good-vs-Medium Separators

| feature | robust medium-good delta | robust bad-medium delta |
|---|---:|---:|
| `flatline_frac` | -4.999 | 0.000 |
| `clipping_frac` | -1.111 | -0.111 |
| `contact_loss_proxy` | -1.000 | 0.000 |
| `prominence_p75` | -0.969 | -1.029 |
| `ptst_unreliable_proxy` | 0.914 | 1.938 |
| `low_amp_frac` | -0.908 | -0.354 |
| `qrs_reliable_proxy` | -0.872 | -1.012 |
| `nonqrs_energy_ratio` | 0.849 | 1.551 |

## Generator Implication

- Medium should be generated as **QRS usable but measurement-details unreliable**, not as weaker bad or stronger good.
- For medium, emphasize P/T/ST ambiguity, non-QRS energy, local derivative spikes, baseline micro-steps, and mild contact/ramp events while preserving QRS timing/prominence.
- For bad, emphasize QRS-confounding pseudo-peaks/contact events; if bad pressure is increased through flatline/low-amplitude alone, medium collapses.
- Next grid selection should penalize rows where medium recall only improves by sacrificing bad recall, and it should retain an explicit medium-cluster geometry score.

## Visual Artifacts

- PCA by class: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_medium_cluster_analysis_10s_2026_06_03\visuals\but_pca_by_class.png`
- Key feature boxplots: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_medium_cluster_analysis_10s_2026_06_03\visuals\but_key_feature_boxplots.png`
- Pairwise feature boundaries: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_medium_cluster_analysis_10s_2026_06_03\visuals\pairwise_feature_boundaries.png`
- Medium feature importance: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_medium_cluster_analysis_10s_2026_06_03\visuals\medium_vs_rest_importance.png`
- Synthetic medium alignment: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_medium_cluster_analysis_10s_2026_06_03\visuals\synthetic_medium_alignment.png`
- Current grid trade-off: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_medium_cluster_analysis_10s_2026_06_03\visuals\current_grid_medium_bad_tradeoff.png`

## Caveat

This is descriptive morphology analysis. It explains the label geometry and generator failure mode, but it does not by itself prove a new synthetic rule will improve BUT test performance.
