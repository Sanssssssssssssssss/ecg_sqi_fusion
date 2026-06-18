# Hard-Feature Waveform Transformer Exploration Update - 2026-06-18

Mirror of the report in:

`E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\hardfeature_framework_exploration_update_20260618.md`

## Scope

This update records the latest waveform-only Transformer framework exploration for the four original-domain hard features:

- `sqi_basSQI`
- `baseline_step`
- `qrs_band_ratio`
- `detector_agreement`

The models still use waveform-only inference. Teacher/SQI targets are training-time supervision and diagnostics only. Original BUT remains report-only.

Machine-readable summaries:

- `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\hardfeature_framework_recovery_20260618.csv`
- `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_node_ladder_tuning_10s_2026_06_08\analysis\good_medium_geometry_repair\hardfeature_framework_bucket_metrics_20260618.csv`

## Prior Frontier

The current waveform-only frontier from the unified checkpoint panel is:

| candidate | original_test_all_10s+ acc | note |
|---|---:|---|
| `featurefirst_top20_qrsbase_artbad_dualcoreout_recall_a050_badcal` | 0.8641 | Current best waveform-only panel result; good/medium/bad recall 0.8896/0.8644/0.6350, bad_core 0.9664, bad_outlier_stress 0.5000. |
| `featurefirst_top20_qrsbase_artbad_dualcoreout_recall_a050` | 0.8637 | Same checkpoint before train/val bad-threshold calibration; bad recall 0.6277. |
| `featurefirst_wavecomp_record111physio_guard_a050` | 0.8601 | Prior best report; bad recall 0.6472 and bad_outlier_stress recall 0.5411. |
| `featurefirst_wavecomp_record111physio_slowlr_warm_a050` | 0.8428 | Strong medium/bad balance, but good recall falls to 0.7574. |

The new candidates below were designed to preserve good recall while improving hard-feature learning. The dual-core/outlier recall candidate is a small but real improvement over the previous waveform-only frontier, although it is still far from the 0.90 target.

## Original Test Buckets

Rows are original BUT report-only. Selection still uses synthetic/node diagnostics, not original BUT.

| candidate | acc | good | medium | bad | good->medium | medium->good | bad_outlier |
|---|---:|---:|---:|---:|---:|---:|---:|
| `featurefirst_top20_qrsbase_artbad_dualcoreout_recall_a050_badcal` | 0.8641 | 0.8896 | 0.8644 | 0.6350 | 397 | 424 | 0.5000 |
| `featurefirst_top20_qrsbase_artbad_dualcoreout_recall_a050` | 0.8637 | 0.8896 | 0.8644 | 0.6277 | 397 | 424 | 0.4897 |
| `featurefirst_top20_qrsbase_dualcoreout_v5badbranch_preserveartifact_balanced_a050_badcal` | 0.8625 | 0.8896 | 0.8656 | 0.5888 | 395 | 426 | 0.4418 |
| `featurefirst_wavecomp_record111physio_slowlr_warm_a050` | 0.8428 | 0.7574 | 0.9311 | 0.6472 | 881 | 134 | 0.5616 |
| `record111_wavecomp_warm_stopgrad_balance_a050_badcal` | 0.8221 | 0.9019 | 0.7826 | 0.5401 | 350 | 777 | 0.4384 |
| `record111_wavecomp_warm_stopgrad_balance_a050` | 0.8203 | 0.9019 | 0.7838 | 0.4915 | 350 | 779 | 0.3801 |
| `record111_detbase_frozenbridge_artifact_a050_badcal` | 0.8149 | 0.9003 | 0.7991 | 0.2287 | 358 | 836 | 0.1884 |
| `record111_detbase_frozenbridge_artifact_a050` | 0.8147 | 0.9003 | 0.7994 | 0.2214 | 358 | 836 | 0.1781 |
| `record111_wavecomp_warm_finetune_balance_a050` | 0.8105 | 0.9071 | 0.7623 | 0.4745 | 330 | 864 | 0.3253 |
| `record111_wavecomp_warm_finetune_balance_a050_badcal` | 0.8114 | 0.9071 | 0.7621 | 0.4939 | 330 | 860 | 0.3527 |
| `record111_detbase_featpre_stopgrad_goodprotect_a050` | 0.8064 | 0.9074 | 0.7704 | 0.2993 | 326 | 947 | 0.0308 |
| `record111_detbase_frozenbridge_balance_a050_badcal` | 0.8052 | 0.9025 | 0.7700 | 0.3236 | 330 | 920 | 0.0479 |
| `record111_detbase_frozenbridge_balance_a050` | 0.8037 | 0.9071 | 0.7763 | 0.1825 | 330 | 963 | 0.0103 |
| `record111_detbase_frozenbridge_goodprotect_a050` | 0.8002 | 0.9096 | 0.7653 | 0.2068 | 322 | 1011 | 0.0103 |
| `record111_detbase_frozenbridge_goodprotect_a050_badcal` | 0.7960 | 0.9055 | 0.7501 | 0.3212 | 318 | 974 | 0.0445 |
| `record111_detbase_featpre_stopgrad_goodprotect_a050_badcal` | 0.7304 | 0.8154 | 0.6853 | 0.4647 | 283 | 506 | 0.2466 |
| `record111_wavecomp_warm_finetune_goodprotect_a050` | 0.8057 | 0.9132 | 0.7526 | 0.4258 | 309 | 939 | 0.2705 |
| `record111_primctx_goodprotect_a050` | 0.7951 | 0.9184 | 0.7352 | 0.3479 | 293 | 1052 | 0.2603 |
| `record111_detbase_featpre_goodprotect_a050` | 0.7897 | 0.9181 | 0.7341 | 0.2506 | 291 | 1150 | 0.0103 |
| `record111_detbase_goodprotect_a050` | 0.7517 | 0.9003 | 0.6672 | 0.3455 | 335 | 1231 | 0.0788 |
| `record111_detbase_hardaux_finetune_goodprotect_a050` | 0.7269 | 0.8937 | 0.6301 | 0.2920 | 308 | 1199 | 0.0925 |
| `record111_detbase_hardaux_goodprotect_a050` | 0.5944 | 0.6514 | 0.5635 | 0.4234 | 455 | 590 | 0.3562 |

## Feature Recovery on Original All 10s+

Values are Pearson correlations from each candidate's `feat_orig.csv`.

| candidate | basSQI | baseline | qrs_band | detector | qrs_vis | flatline | detail | bSQI | pc1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `featurefirst_top20_qrsbase_artbad_dualcoreout_recall_a050` | 0.289 | 0.459 | 0.663 | 0.310 | 0.476 | 0.796 | 0.947 | 0.911 | 0.969 |
| `featurefirst_top20_qrsbase_dualcoreout_v5badbranch_preserveartifact_balanced_a050` | 0.289 | 0.459 | 0.663 | 0.310 | 0.476 | 0.796 | 0.947 | 0.911 | 0.969 |
| `featurefirst_wavecomp_record111physio_guard_a050` | -0.141 | -0.226 | -0.404 | -0.126 | 0.463 | 0.172 | 0.922 | 0.927 | 0.962 |
| `featurefirst_wavecomp_record111physio_slowlr_warm_a050` | -0.056 | -0.078 | 0.485 | -0.052 | 0.500 | 0.520 | 0.923 | 0.928 | 0.972 |
| `physiofact_detbase_token_balanced_a050` | 0.228 | 0.429 | 0.667 | 0.316 | 0.420 | 0.779 | 0.943 | 0.905 | 0.960 |
| `record111_detbase_frozenbridge_goodprotect_a050` | 0.228 | 0.429 | 0.667 | 0.316 | 0.420 | 0.779 | 0.943 | 0.905 | 0.960 |
| `record111_detbase_frozenbridge_balance_a050` | 0.228 | 0.429 | 0.667 | 0.316 | 0.420 | 0.779 | 0.943 | 0.905 | 0.960 |
| `record111_detbase_frozenbridge_artifact_a050` | 0.228 | 0.429 | 0.667 | 0.316 | 0.420 | 0.779 | 0.943 | 0.905 | 0.960 |
| `record111_detbase_goodprotect_a050` | -0.213 | -0.350 | -0.669 | 0.055 | 0.330 | 0.284 | 0.923 | 0.907 | 0.956 |
| `record111_detbase_hardaux_goodprotect_a050` | 0.333 | 0.105 | -0.044 | 0.322 | 0.512 | 0.556 | 0.562 | 0.794 | 0.640 |
| `record111_primctx_goodprotect_a050` | -0.142 | -0.291 | -0.546 | -0.115 | 0.429 | 0.181 | 0.886 | 0.923 | 0.922 |
| `record111_detbase_hardaux_finetune_goodprotect_a050` | -0.127 | -0.266 | -0.512 | 0.196 | 0.363 | 0.450 | 0.873 | 0.905 | 0.919 |
| `record111_detbase_featpre_goodprotect_a050` | -0.186 | -0.213 | 0.627 | 0.218 | 0.419 | 0.620 | 0.941 | 0.913 | 0.970 |
| `record111_wavecomp_warm_finetune_goodprotect_a050` | -0.155 | -0.226 | -0.382 | -0.111 | 0.424 | 0.196 | 0.929 | 0.924 | 0.974 |
| `record111_wavecomp_warm_finetune_balance_a050` | -0.152 | -0.232 | -0.408 | -0.116 | 0.430 | 0.190 | 0.925 | 0.925 | 0.970 |
| `record111_detbase_featpre_stopgrad_goodprotect_a050` | -0.093 | -0.109 | 0.695 | 0.212 | 0.449 | 0.654 | 0.937 | 0.918 | 0.967 |
| `record111_wavecomp_warm_stopgrad_balance_a050` | -0.164 | -0.250 | -0.412 | -0.099 | 0.425 | 0.177 | 0.926 | 0.925 | 0.972 |

## Findings

1. Good recall can be protected, but current protection trades away medium and bad.

`record111_wavecomp_warm_finetune_balance_a050` is the best new "protect good" candidate: original_test good recall is 0.9071, compared with 0.7574 for the slow-LR warm frontier. However, medium drops to 0.7623 and bad drops to 0.4745. This means the model is not yet learning a stable shared boundary; it is moving the decision surface.

2. The detector/baseline token family can learn the hard features, but not under the full Record111 good-protect objective.

`physiofact_detbase_token_balanced_a050` proves the architecture can recover waveform-computable hard features better: `baseline_step=0.429`, `qrs_band_ratio=0.667`, `detector_agreement=0.316`, `flatline_ratio=0.779`. After adding Record111/good-protect training, most of that recovery collapses unless the class surface also collapses.

3. Very strong hard-feature auxiliary loss is not enough.

`record111_detbase_hardaux_goodprotect_a050` improves `sqi_basSQI`, `detector_agreement`, `qrs_visibility`, and `flatline_ratio`, but original_test acc falls to 0.5944. It also damages `non_qrs_diff_p95` and `pc1`. This suggests hard-feature learning is competing with class geometry instead of providing a reusable representation.

4. Fine-tuning repairs class behavior by erasing hard-feature gains.

`record111_detbase_hardaux_finetune_goodprotect_a050` raises synthetic/node behavior again, but original hard-feature recovery moves back toward the old failure mode: `sqi_basSQI=-0.127`, `baseline_step=-0.266`, `qrs_band_ratio=-0.512`.

5. The current bottleneck is not only loss weight. It is representation routing inside the Transformer.

The best current evidence is a conflict:

- Detbase/warm features can learn bands, flatline, detail, and part of detector agreement.
- Good-protect classification can preserve good recall.
- The same single shared token/classification path does not preserve both at once.

6. Stop-gradient late fusion is a partial positive result, not the final fix.

`record111_wavecomp_warm_stopgrad_balance_a050` improves the previous warm-balance original_test bucket from 0.8114 to 0.8221 with bad calibration, while preserving good recall at 0.9019 and raising bad recall to 0.5401. This confirms that preventing classification gradients from rewriting the auxiliary path helps stability. However, its hard-feature recovery remains poor for `sqi_basSQI`, `baseline_step`, `qrs_band_ratio`, and `detector_agreement`.

`record111_detbase_featpre_stopgrad_goodprotect_a050` better preserves the detbase feature axes (`qrs_band_ratio=0.695`, `flatline_ratio=0.654`, `qrs_visibility=0.449`), but original_test bad_outlier remains weak and bad-calibration trades away too much good/medium. The remaining unsolved part is the domain-direction mismatch for `sqi_basSQI` and `baseline_step`, not just gradient interference.

7. Frozen detbase bridges prove feature preservation alone is not enough.

`record111_detbase_frozenbridge_goodprotect_a050` and `record111_detbase_frozenbridge_balance_a050` freeze the detbase feature backbone and only train a small bridge/classification surface. This preserves the detbase hard-feature recovery (`sqi_basSQI=0.228`, `baseline_step=0.429`, `qrs_band_ratio=0.667`, `detector_agreement=0.316`), and synthetic/node behavior remains strong. However, original_test bad recall remains low (0.2068/0.1825 raw; 0.3212/0.3236 after bad calibration), and bad_outlier_stress is almost absent (0.0103 raw, about 0.045-0.048 calibrated). The learned feature map is useful, but the current bridge does not know how to apply it to Record111-style bad outliers.

8. The artifact residual helped medium, but still did not solve bad.

`record111_detbase_frozenbridge_artifact_a050` improves original_test acc to 0.8149 with bad calibration and gives the best medium recall among the frozen-bridge variants (about 0.799). But bad_core falls to 0.3277 and bad_outlier_stress is only 0.1884. This says the added artifact residual is acting more like a medium/detail support path than a robust bad-stress detector. The bad branch needs a more local event/morphology objective, not simply a stronger residual logit.

9. The best current waveform-only result comes from split bad evidence, not from the newer detrawbeat/v5 feature learners.

The unified report-only checkpoint panel found `featurefirst_top20_qrsbase_artbad_dualcoreout_recall_a050_badcal` as the current best: original_test_all_10s+ acc 0.8641, good/medium/bad recall 0.8896/0.8644/0.6350, bad_core 0.9664, and bad_outlier_stress 0.5000. Its original_test confusion is `[[3238, 397, 5], [424, 3826, 176], [105, 48, 258]]`, so reaching 0.90 would require recovering about 307 more windows. The remaining errors are mixed: good/medium boundary plus bad->good, not one isolated class.

The detrawbeat/v5 sparse candidates proved that bad_core can be learned (`bad_core_nearboundary=1.0`), but they dropped original_test acc to about 0.77-0.78 because medium recall fell to roughly 0.70-0.74. They are useful as feature-learning evidence, not as the frontier classifier.

## Next Architecture Direction

Do not continue blind aux-weight sweeps. The next clean experiment should decouple the representation paths while preserving waveform-only inference:

1. Use a detector/baseline/detail token branch trained strongly for waveform-computable SQI recovery.
2. Use a separate class token branch trained for good/medium/bad.
3. Fuse only a compact predicted-SQI state into the class token late, with stop-gradient or low-gradient consistency so classification fine-tuning cannot erase SQI recovery.
4. Add explicit bad-stress specificity after that fusion, but make it local/event-calibrated and keep the dual-core/outlier split that currently gives the best frontier.
5. Keep `sqi_basSQI`, `baseline_step`, `qrs_band_ratio`, and `detector_agreement` as the main diagnostic gates before trusting another original report.

In short: the useful next move is a decoupled SQI-token Transformer that starts from the dual-core/outlier frontier, protects good/medium, and adds only a narrow local bad-outlier event branch. Broad v5 bad-stress training learns bad_core but spends too much medium recall.
