# Waveform-Only Transformer Research Summary 2026-06-14

Goal: move away from tabular SQI/geometry input and test whether a Transformer-family waveform model can learn the same ECG structure directly from waveform channels.  SQI/geometry columns are used only as auxiliary prediction targets, never as classifier inputs.

## Main Finding

The best Transformer-only direction is:

`ConvSubsample Transformer + robust3 waveform channels + auxiliary 47-feature prediction + ECG-style augmentation + focal loss`.

It learns the original validation split very well, but held-out original test remains limited by record/domain shift:

- Best augmented ConvSubsample validation: acc `0.9801`, good/medium/bad recall `0.9814/0.9619/0.9880`.
- Held-out original test for that checkpoint: acc `0.8110`, macro-F1 `0.7304`, good/medium/bad recall `0.9607/0.7259/0.4015`.
- Bad-threshold calibration improves bad recall but costs medium/good: acc `0.7997`, good/medium/bad `0.9516/0.6805/0.7372`.
- Strict val-selected Transformer ensemble reaches about acc `0.8131`; an oracle-sorted diagnostic showed a possible `0.8584`, but that is not a valid selection result.

## Why It Is Hard

The BUT protocol split is record-domain split, not random-window split.

- Train: mostly records `100001` and `105001`.
- Validation: records `103001/103002/103003/114001/126001`.
- Test: mostly record `111001`, plus `122001/125001`.

Bad labels differ sharply by split:

- Train bad is mostly `right_bad_island` from `105001`.
- Test bad is `near_bad_boundary` plus `outlier_low_confidence`, mainly from `111001`.
- The models consistently learn `bad_core/near-boundary` well, but `bad_outlier_stress` remains the hard part.

## Experiments Run

1. Synthetic-only Transformer waveform training:
   - Conformer, MultiPatch Transformer, ConvSubsample Transformer.
   - Best held-out original test was around `0.828`, but bad outlier remained weak.

2. Original train/val/test Transformer adaptation:
   - Original train labels used, val for selection, test held out.
   - ConvSubsample val reached `0.9793` with good/medium/bad `0.9835/0.9238/1.0000`.
   - Test reached `0.8077` raw, or `0.7855` with badcal but bad recall `0.8151`.

3. Synthetic pretrain plus original fine-tune:
   - Did not improve transfer; synthetic initialization pulled the model toward synthetic geometry and did not solve bad outlier.

4. ECG-style augmentation:
   - Random shift, amplitude scaling, baseline wander, noise, short mask, slope drift.
   - Strongest valid single-model direction so far.

5. Bad-outlier-specific augmentation:
   - Simulated longer dropouts/contact/baseline/high-frequency stress for bad only.
   - Did not help; it damaged good/medium and still failed to cover `bad_outlier_stress`.

## Current Best Transformer Candidates

- `aug_convtx_balanced_focal`
  - Checkpoint: `outputs/.../runs/waveform_transformer_augmented_original/N17043_gm_probe/aug_convtx_balanced_focal/ckpt_best.pt`
  - Best validation epoch: `7`
  - Original test raw: acc `0.8110`, good/medium/bad `0.9607/0.7259/0.4015`
  - Original test badcal: acc `0.7997`, good/medium/bad `0.9516/0.6805/0.7372`

- `orig_convtx_robust3_aux`
  - Original test raw: acc `0.8077`, good/medium/bad `0.9835/0.6792/0.6350`
  - Original test badcal: acc `0.7855`, good/medium/bad `0.9701/0.6310/0.8151`

## Recommended Next Step

Do not keep sweeping broad weights.  The next clean Transformer experiment should be record-domain robust training:

- keep ConvSubsample Transformer as the base;
- train with stronger record-style augmentation, but only if it preserves good/medium;
- add validation objective that explicitly penalizes `medium->good` and `bad_outlier->medium`;
- consider pretraining on synthetic plus unlabeled original waveform reconstruction/contrastive objectives, then supervised fine-tuning.

The honest status is: waveform Transformer is now viable and much better understood, but it has not yet matched the 47-feature tabular model.  The gap is not architecture alone; it is the held-out `111001` domain and bad-outlier stress distribution.
