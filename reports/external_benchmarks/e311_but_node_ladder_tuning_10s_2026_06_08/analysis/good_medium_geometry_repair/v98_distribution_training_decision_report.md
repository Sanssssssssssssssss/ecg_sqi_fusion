# V98 Distribution / Training Decision Snapshot

Created: 2026-06-24

## Current Accepted Roles

- Training high-confidence PTB protocol: `protocol_v95_gm_boundary_clean_drop4_from_v94_s20260710`
  - Drops four visually/feature-ambiguous good/medium subtype buckets.
  - Best PTB self-test normal waveform model: `E1_query_only`
  - PTB test acc `0.962672`, macro-F1 `0.955186`
  - Recalls: good `0.940741`, medium `0.925676`, bad `1.000000`

- BUT fixed protocol: `margin_ge_5s_keep_outlier_drop_mediumlike_bad_seed20260623_v37subtype_fixed`
  - Best BUT-only from-scratch model: `E0_noquery_nohi_nolocal_noart`
  - BUT test acc `0.932772`, macro-F1 `0.938225`
  - Recalls: good `0.931848`, medium `0.919110`, bad `0.989796`
  - BUT all acc `0.941190`, macro-F1 `0.947771`

- Natural-prior distribution visualization protocol: `protocol_v98_metric_hybrid_transport_pc1500_s20260722`
  - Uses v96 good/medium plus per-bad-subtype best rows from v96/v97.
  - Intended for distribution and waveform evidence, not as the current 3-class training main set.
  - V98 PTB self-test did not meet the 96 target:
    - `E0_noquery_nohi_nolocal_noart` test acc `0.843982`, macro-F1 `0.846075`
    - `E1_query_only_subtype_aux` test acc `0.835067`, macro-F1 `0.836986`

## Distribution Findings

V98 improves bad distribution metrics over v96 in aggregate:

| Run | Class | RBF-MMD | Sliced-Wasserstein | Quantile loss | PCA overlap |
|---|---|---:|---:|---:|---:|
| v96 | bad | 0.8535 | 8.3494 | 5.7075 | 0.0158 |
| v98 | bad | 0.8258 | 5.8961 | 3.9272 | 0.0737 |

Largest clear win:

| Subtype | v96 RBF-MMD | v98 RBF-MMD | v96 PCA overlap | v98 PCA overlap |
|---|---:|---:|---:|---:|
| bad_contact_reset_flatline | 0.5682 | 0.1630 | 0.0909 | 0.4960 |

Remaining gaps:

- `detector_agreement` and `sqi_iSQI` remain nearly discrete and domain-separable.
- `wavelet_e*`, `hjorth_mobility`, `sample_entropy_proxy`, and `sqi_sSQI` still separate synthetic bad from BUT bad.
- Domain classifier AUC remains `1.0`, so the distribution is visually better but not statistically matched.

## Interpretation

The current evidence supports a two-protocol strategy:

- Use `v95` as the high-confidence training protocol because it reaches PTB 96 and has stable good/medium/bad learning.
- Use `v98` as the natural-prior distribution/waveform visualization protocol because it preserves more BUT-like bad morphology and shows the remaining domain gap honestly.

Do not use v98 as the primary 3-class training set yet. Its lower self-test indicates that the natural-prior subtype mixture includes label-boundary ambiguity that should either be handled as confidence/auxiliary supervision or split into diagnostic/stress buckets.

## Key Outputs

- V95 PTB model report:
  `reports/.../event_factorized_sqi_conformer/event_factorized_single_domain_selftest_v95_gm_boundary_clean_ptb_report.md`
- V96 BUT seed-sweep output:
  `outputs/.../event_factorized_single_domain_selftest_v96_but_seed_sweep/single_domain_selftest_summary.csv`
- V98 distribution report:
  `reports/.../v98_metric_hybrid_distribution_transport/v81_distribution_transport_report.md`
- V98 shared PCA:
  `reports/.../v98_metric_hybrid_distribution_transport/v98_shared_pca.png`
- V98 bad waveform examples:
  `reports/.../v98_metric_hybrid_distribution_transport/v98_bad_individual_waveform_examples.png`

