# v115 Support-Aware Distribution Repair Summary

## What Changed

v115 separates two claims that were previously easy to mix:

- Clean-only synthetic line: PTB clean physiology + BUT good style + non-clean feature targets + mechanistic artifact generator. It does not use BUT medium/bad waveform donor rows.
- Semi-synthetic repair line: controlled BUT train-only native support may be included. This is target-domain-informed distribution repair, not a clean-only generation claim.

The selector was changed from row-distance selection to set-level distribution optimization. The GPU path uses torch/CUDA random Fourier feature MMD proposals, then exact v110 metrics are recomputed after selection.

## Key Result

Strict native-fraction grid, seed `20260843`, selected 300 rows/class:

| line | actual native | all-label MMD | good MMD | medium MMD | bad MMD | bad PCA overlap |
|---|---:|---:|---:|---:|---:|---:|
| clean-only 0% | 0.00 | 0.2735 | 0.3534 | 0.2990 | 0.6544 | 0.0066 |
| semi-synth 10% | 0.086 | 0.2384 | 0.2897 | 0.2406 | 0.6221 | 0.0601 |
| semi-synth 25% | 0.250 | 0.1736 | 0.2010 | 0.1626 | 0.3964 | 0.2278 |
| semi-synth 55% | 0.550 | 0.0870 | 0.0728 | 0.0558 | 0.0997 | 0.5084 |

Interpretation: clean-only still cannot reconstruct BUT-like bad support. Adding controlled BUT train-only support makes the distribution match improve smoothly and substantially, especially for bad.

## Support Audit Caveat

For bad, q95 coverage can be misleading because the BUT-vs-BUT q95 floor is inflated by sparse/extreme bad targets. q90 coverage remains low in clean-only runs, while v110 MMD and PCA overlap still show the clean-only bad support gap.

## Important Paths

- Runner: `outputs/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/run_v115_support_aware_distribution_repair.py`
- Strict fraction-grid report: `E:/GPTProject2/ecg_keep_20260528_172844/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/v115_support_aware_distribution_repair/s20260843`
- Clean-only set-level run: `E:/GPTProject2/ecg_keep_20260528_172844/reports/external_benchmarks/e311_but_node_ladder_tuning_10s_2026_06_08/analysis/good_medium_geometry_repair/v115_support_aware_distribution_repair/s20260841`

## Current Decision

Do not present the low-MMD 55% line as PTB clean-only generation. The honest conclusion is:

1. Clean-only PTB support remains insufficient for BUT bad morphology.
2. Controlled train-domain support can repair the distribution and reach the desired MMD regime.
3. The next clean-only improvement must target generator support, especially structured bad regimes, rather than more selector tuning.
