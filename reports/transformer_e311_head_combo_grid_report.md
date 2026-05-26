# E3.11 Head Combination Grid

This grid searches task-head combinations on the E3.11f mainline data, without changing the backbone or labels.
The fixed strong recipe is CLS pooling, positional embedding, raw input, D1 warm-start, and validation-accuracy checkpoint selection.

Rationale references used for the grid design:

- MTECG: ECG segment tokens with learnable positional embeddings and masked-autoencoder style representation learning: https://arxiv.org/abs/2309.07136
- UniTS: time-series models can share parameters across classification, imputation, and related tasks: https://github.com/mims-harvard/UniTS
- SwinDAE: ECG quality assessment can benefit from pairing Transformer features with denoising-autoencoder objectives: https://pubmed.ncbi.nlm.nih.gov/37698969/

Historical anchors:

- E3.10 best single visual: `0.9402`
- E3.11f best before head-combo grid: `0.9464` (`r3_lr625_seed1`)
- E3.11f stable basin: `lr=5.75e-5`, dropout `0.10`, mean about `0.9423` across prior seeds

## Top Runs

| Rank | Run | Family | Test Acc | Recall G/M/B | Head/Loss Summary |
| ---: | --- | --- | ---: | --- | --- |
|  | pending | pending |  |  |  |

## Family Summary

| Family | N | Mean Acc | Std | Best Acc | Best Run |
| --- | ---: | ---: | ---: | ---: | --- |
| pending |  |  |  |  |  |

## Live Decision Rules

- If SNR-only remains best, keep the model simple and spend future effort on data/source audit.
- If ordinal improves medium recall without hurting bad recall, expand ordinal around `lambda_ord=0.03-0.05`.
- If local mask helps, keep it at low weight only; the current target is injected-noise envelope, so high weights may overfit synthetic placement.
- If denoise/level auxiliary improves classification by at least `0.003`, rerun the best denoise setting across multiple seeds.
- If noise-type head is flat or negative, drop it; E3.11f uses only `em/ma/mix`, so texture supervision may be a distraction.
