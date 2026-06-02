# Reviewer Analysis Prompt: E3.11f Warm Denoiser Encoder-Head Direction

Please analyze this experiment package as an external reviewer. The goal is not to blindly select the highest accuracy run, but to decide whether the newly observed warm-start denoiser-encoder classification framework is scientifically valid enough to become the next mainline direction.

Important update: the user will not accept a U-Net-only final model. Treat the U-Net encoder-head results as oracle/teacher/mechanism evidence. The next mainline must restore a Transformer classifier, ideally with the warm denoiser latent entering through a controlled SQI token/adapter rather than replacing the Transformer.

## Repository Context

Branch:

`experiment/factorial-local-splits`

Primary report package:

`reports/e311_encoder_head_audit_2026_06_01`

Main summary:

`reports/e311_encoder_head_audit_2026_06_01/README.md`

Transformer re-entry plan package:

`reports/e311_transformer_reentry_2026_06_01/README.md`

Important copied reports:

- `copied_reports/MATURE_DENOISE_BASELINE.md`
- `copied_reports/FULL_JOINT_REPORT.md`
- `copied_reports/CONFLICT_MTL_REPORT.md`
- `copied_reports/CONFLICT_CAP_REPORT.md`
- `copied_reports/FOCUSED_2H_REPORT.md`

Key metrics:

- `metrics/encoder_head_only_run_summary.json`
- `metrics/encoder_head_only_test_report.json`
- `metrics/encoder_head_only_denoise_metrics.json`
- `metrics/dual_branch_encoder_delta_run_summary.json`
- `metrics/dual_branch_encoder_delta_test_report.json`
- `metrics/dual_branch_encoder_delta_denoise_metrics.json`
- `metrics/denoiser_warm_classifier_he_test_report.json`
- `metrics/denoiser_warm_classifier_he_denoise_metrics.json`
- `metrics/he_scratch_test_report.json`
- `metrics/he_scratch_denoise_metrics.json`

Representative figures:

- `figures/mature_baseline_denoise_compact_test.png`
- `figures/encoder_head_only_denoise_compact_test.png`
- `figures/dual_branch_encoder_delta_denoise_compact_test.png`
- `figures/denoiser_warm_balanced_gallery.png`
- `figures/denoiser_warm_worst_residual_gallery.png`
- `figures/conflict_cap_pareto_acc_bad_denoise.png`
- `figures/conflict_cap_gradient_raw_projected_applied.png`

## Data And Split Facts

The current sweep dataset is:

`med6p25_badgap7_badcm0p75`

Full split:

- train: `10935 = 3645 x 3`
- val: `2184 = 728 x 3`
- test: `2202 = 734 x 3`

Exact overlap checks already performed locally:

- `ecg_id`: train/val/test overlap all `0`
- `counterfactual_group`: train/val/test overlap all `0`
- `source_npz_index`: train/val/test overlap all `0`
- `seg_id`: train/val/test overlap all `0`

This reduces the chance of exact split leakage, but does not eliminate synthetic-rule shortcut risk. Each original ECG segment has good/medium/bad counterfactual versions within its split, and labels are generated from SNR/morphology damage rules.

## Main Phenomena To Explain

### 1. Warm denoiser encoder-head is unexpectedly strong

Run:

`encoder_head_only_e8_seed0`

Architecture:

`noisy ECG -> warm mature residual_unet denoiser -> denoiser encoder latent -> MLP classifier`

Initialization:

- denoiser: warm mature denoiser checkpoint
- classifier head: newly initialized MLP
- trainable: denoiser + MLP head
- not a cached-head-only shortcut
- full split was used

Result:

- acc: `0.986376`
- good/medium/bad recall: `0.980926 / 0.979564 / 0.998638`
- denoise_score: `3.611793`
- MSE ratio: `0.068449`
- SNR gain: `+10.206917 dB`

This is currently the best balanced candidate if we require strong classification and strong denoise.

Question:

Is this a valid framework, or is it mainly evidence that the denoiser encoder has learned a near-direct representation of the synthetic label-generation rule?

Current decision constraint:

Do not recommend promoting this exact U-Net encoder-head as mainline. Use it to decide what information the Transformer must receive.

### 2. Highest accuracy waveform joint run is not a true denoiser

Run:

`denoiser_warm_classifier_he / denwarm_clshe_gated_cap0p20_ld7_e40_seed0`

Architecture:

`noisy ECG -> warm denoiser -> denoised waveform -> He-init Transformer classifier -> SQI residual`

Result:

- acc: `0.988193`
- good/medium/bad recall: `0.986376 / 0.980926 / 0.997275`
- denoise_score: `-0.068925`
- SNR gain: `-0.337154 dB`
- SSIM delta: `-0.438981`
- good-class SNR gain: about `-7.16 dB`

Question:

Why can classification get better while denoise metrics get worse? The likely hypothesis is that CE through the denoiser turns it into a discriminative filter rather than a clean ECG reconstructor. Please evaluate whether this explanation fits the evidence.

### 3. Scratch training can classify but fails denoise

Run:

`he_scratch_A_decay_cap0p10_e50_seed0`

Initialization:

- denoiser: He/Kaiming scratch
- classifier: He/Kaiming scratch
- SQI head: scratch

Result:

- acc: `0.973660`
- good/medium/bad recall: `0.956403 / 0.974114 / 0.990463`
- denoise_score: `-1.658656`
- SNR gain: about `-4.22 dB`

Question:

Why does non-warm training learn classification but not denoise? Possible explanations:

- classification labels are easier than clean waveform reconstruction;
- bad/medium labels are strongly tied to visible synthetic SNR/morphology artifacts;
- CE gradients dominate the shared path and push away from clean denoise;
- denoise from scratch has a hard local minimum unless pre-trained with morphology-aware loss;
- the model can solve class boundaries without learning clinically faithful denoise.

Please decide which explanation is most consistent with the results.

### 4. Dual branch does not beat encoder-head-only

Run:

`dual_branch_encoder_delta_e8_seed0`

Architecture:

`noisy ECG -> warm denoiser -> denoised waveform -> warm Transformer classifier`

plus

`denoiser encoder latent -> gated residual delta on logits`

Result:

- final acc: `0.962761`
- base waveform classifier acc at selected checkpoint: `0.722525`
- good/medium/bad recall: `0.952316 / 0.944142 / 0.991826`
- denoise_score: `3.165726`
- SNR gain: `+8.774765 dB`

Question:

Why is this weaker than encoder-head-only? Does the waveform classifier drift or collapse when the denoiser keeps changing? Is the encoder delta repairing an unstable base classifier rather than providing a clean SQI correction?

Follow-up implementation now being tested:

- `transformer_film_sqi`: denoised ECG goes through `MTLTransformerPTBXL`; warm denoiser latent applies zero-init FiLM-style residual modulation to the Transformer pooled feature.
- `transformer_cross_sqi`: denoised ECG goes through `MTLTransformerPTBXL`; Transformer CLS pooled feature performs small cross-attention over the warm denoiser latent.
- `transformer_sqi_token_prefix`: warm denoiser latent is projected as an SQI token prepended after CLS before Transformer encoder blocks.

All three are experiment-only and keep final logits on the Transformer path.

### 5. PCGrad/cap controlled conflicts but did not solve the architecture

Best PCGrad/cap run:

`cap_A_decay_0p10_A_ce_primary_pcgrad_cap_decay_cap0p1_ab490fa6`

Result:

- acc: `0.972752`
- good/medium/bad recall: `0.950954 / 0.974114 / 0.993188`
- denoise_score: `2.282515`
- applied aux ratio controlled at about `0.10`
- applied cosine positive

Question:

Does PCGrad/cap prove that loss conflict is real and controllable, even if warm denoiser pretraining is a stronger practical solution?

## Candidate Mainline Direction

The user rejected a U-Net-only mainline. The candidate mainline must be reframed as:

1. Train a morphology-aware denoiser first.
2. Feed denoised ECG into a Transformer classifier.
3. Inject denoiser-derived SQI/morphology latent into the Transformer through a small controlled adapter.
4. Keep waveform denoise output as a first-class output.
5. Avoid letting CE destroy denoise morphology.

Please evaluate whether this is a publishable framing:

`pretrain denoiser -> provide denoised waveform + SQI latent -> Transformer classifies quality`

This may be more principled than simultaneous CE + denoise training because it turns warm-start into a curriculum / task-conflict solution.

## What Not To Confuse

- The `encoder_head_only` run is warm-started, not scratch.
- The `encoder_head_only` run is not Transformer-based and must not be treated as final mainline.
- The `dual_branch` run uses a warm Transformer classifier, but still trains full split.
- `classifier_shared` in older PCGrad reports means the Transformer classifier encoder parameter group, not the denoiser encoder head architecture.
- The `0.988193` run has the highest accuracy but weak denoise fidelity; do not recommend it as mainline without caveats.
- Exact split leakage checks were clean, but synthetic-rule shortcut risk remains.

## Required Reviewer Output

Please answer:

1. Is warm denoiser encoder-head currently the strongest balanced oracle/teacher framework?
2. Which Transformer re-entry method is most scientifically defensible: SQI token prefix, FiLM adapter, cross-attention adapter, or residual logit delta?
3. Why does warm start solve the multi-task conflict better than scratch or PCGrad-only training?
4. Why is non-warm/scratch denoise so poor while classification remains high?
5. Is the high bad recall a sign of robust quality understanding or an artifact of SNR/morphology label construction?
6. Which final audits are mandatory before promotion?
7. Can a fully Transformer denoiser/classifier path plausibly replace the U-Net warm denoiser, or should U-Net remain as a teacher/denoiser while Transformer owns classification?

Recommended mandatory audits:

- Freeze mature denoiser and train only the encoder MLP head.
- Train `encoder_head_only` with `detach_encoder_features=True` so CE cannot modify denoiser.
- Train scratch encoder-head as a negative control.
- Cross-gap evaluation on gap5/gap6.5/gap7.
- Noisy-only, denoised-only, residual-only, clean-only classifier probes.
- Patient-level split audit if PTB-XL patient metadata can be mapped.
- Seed repeat for the encoder-head framework.

Promotion rule proposal:

- Keep denoise_score clearly positive, ideally `>= 3.0`.
- Keep acc above the mature baseline by a large margin, ideally `>= 0.975`.
- Ensure warm encoder-head still wins when CE is detached or denoiser is frozen.
- Ensure no residual-only probe can match the full model too closely.
- Demonstrate cross-gap stability.
