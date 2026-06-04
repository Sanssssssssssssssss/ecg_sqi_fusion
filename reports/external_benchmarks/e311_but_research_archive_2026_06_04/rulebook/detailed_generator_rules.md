# Detailed Generator Rules

## 1. b10 / Bad-Boundary Wearable Rules

Purpose: make synthetic bad look like real wearable/BUT bad instead of simply low SNR.

Bad artifacts:
- contact loss and flat segments;
- baseline step/dropout;
- clipping and low-amplitude compression;
- spurious QRS-like bursts;
- mixed wearable motion/HF/baseline noise.

Medium rule at this stage was still too much like intermediate severity. b10 worked because it gave bad a realistic wearable flavor, but later analysis showed medium was still under-modeled.

## 2. Medium/Bad and Medium-Mixture Rules

Purpose: recover BUT class 2 by separating "detail unreliable" from "unusable".

Medium artifacts:
- QRS is mostly preserved;
- P/T/ST and non-QRS regions receive local unreliability;
- short baseline steps/ramps;
- inverted or pseudo local events;
- small local contact disturbances.

Bad artifacts:
- QRS-confounding pseudo-peaks;
- stronger contact or baseline failures;
- motion/HF bursts.

Finding: P/T/ST/local-detail events help medium, but if bad pressure is too weak, bad recall collapses; if bad pressure is too strong, medium is swallowed.

## 3. Morph Sweet / V2 / V3 Rules

Purpose: tune by visual morphology rather than SNR.

Rule axes:
- medium QRS preserve level;
- P/T/ST instability strength;
- local baseline steps/ramps;
- pseudo-peak density;
- contact-loss severity;
- good mild wearable overlap.

Finding: local morphology alone is not enough. Medium and bad cannot be described by one monotonic artifact strength axis.

## 4. Morphology-Guided Grid

Purpose: use BUT feature profiles and previous grid rows as priors.

Rule families:
- medium rescue: QRS visible, local detail unreliable;
- bad rescue: QRS-confounding/contact events, not pure flatline;
- coexistence anchors: interpolate b10, s02, mix05, good-not-pristine;
- negative controls: SNR/flatline-heavy rules.

Best anchor: h_bad_rescue_05 with acc 0.8229, balanced 0.8177, macro-F1 0.7454, recalls 0.887/0.773/0.793.

## 5. Fatal-OR Logic

Purpose: encode the user insight that one fatal dimension failing can make a strip bad.

Boundary:
- good = AND(all critical dimensions good);
- bad = OR(any fatal dimension fails hard);
- medium = independent QRS-usable/detail-unreliable cluster.

Problem: applying averaged fatal pressure to all bad examples still creates an unrealistic single bad distribution.

## 6. Sample-Level Fatal Subtype OR

Purpose: make each bad sample trigger one or two fatal subtypes instead of every bad artifact at once.

Bad subtypes:
- qrs_confound: attenuate/confuse QRS and insert QRS-like distractors;
- contact_flat: flat/contact-loss spans;
- motion_burst: burst/HF/wander motion noise;
- morph_break: local morphology damage in QRS/TST/critical regions;
- clipping_lowamp: clipping and amplitude compression;
- baseline_jump: short steps/ramps and baseline discontinuities.

Finding: this is mechanistically plausible but did not beat h_bad_rescue_05. Quick results were better than full; full training improved PTB/denoise but strengthened the wrong synthetic boundary.

## 7. Diagnostic-Usability Latest Hypothesis

Purpose: match BUT visual evidence that some bad examples retain visible QRS but remain globally unreliable.

New bad subtype:
- visible_qrs_global_noise: preserves much of QRS while adding global HF/burst/drift, repeated distractors, and unreliable background.

Boundary:
- good is strict: all critical dimensions clean enough;
- medium is QRS usable with local/detail unreliability;
- bad can be visible-QRS global unusability, contact failure, or QRS-confounding fatal failure.

This runner is intentionally quick/probe only. Full training is disabled by default until quick results beat anchors.
