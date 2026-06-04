# Next Hypotheses

1. **Diagnostic-usability bad, not all-destroyed bad.** BUT class 3 can keep visible QRS while the strip is globally unreliable. Continue testing `visible_qrs_global_noise` and avoid treating all bad examples as flat/QRS-erased.

2. **Medium is an independent cluster.** Keep QRS visible and introduce local detail unreliability. Do not tune medium by simply lowering SNR or weakening bad.

3. **Full training only after quick boundary works.** Prior full confirmations improved PTB and denoise but pulled the classifier toward the synthetic boundary, hurting BUT. Use quick/probe as a boundary discovery stage.

4. **Feature proxy is diagnostic, not decisive.** Morphology distance should explain hypotheses and veto obvious mismatches, but feature_target_score alone should not select models.

5. **Report zero-shot and supervised adaptation separately.** Head-only BUT adaptation can show representation transfer, but it is not strict synthetic zero-shot evidence.
