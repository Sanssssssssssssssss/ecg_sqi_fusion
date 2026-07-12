# Model Catalogue

Four frozen models share one input and output layer. Bundle verification runs
before prediction so a corrupted or substituted artifact is rejected before
deserialization.

| Identifier | Leads | Model family | Classes | Artifact |
|---|---:|---|---|---|
| `12lead-conformer` | 12 | Lead-wise shared Conformer | acceptable / unacceptable | `best_model.pt` |
| `12lead-rbfsvm` | 12 | RBF-SVM over profile-defined SQIs | acceptable / unacceptable | `model.joblib` |
| `singlelead-conformer` | 1 | GM-mechanism Conformer | good / medium / bad | `ckpt_best.pt` |
| `singlelead-rbfsvm` | 1 | RBF-SVM over profile-defined SQIs | good / medium / bad | `model.joblib` |

## Artifact integrity

| Model | SHA-256 |
|---|---|
| 12-lead Conformer | `f08e97226ed0bed5cadcced470708572b32dcc0894e5d842a59866da307a3654` |
| Single-lead Conformer | `17d7dc331de40862943d3f04b372719cdc53194ee0d10d41fc3e239239dc1c7a` |
| 12-lead RBF-SVM | `3d739eac5d378d08c0a47e913979a5f127618fe6a0a868a3095d2cb6a69031c2` |
| Single-lead RBF-SVM | `906ed289c203cf6fab9f85ec4eee0f99f8d07d5f65b8f05e3aa9fc38b1f6e7ee` |

Profiles are hashed separately in `pretrained/inference/manifest.json` because
they define feature order, thresholds, class order, and normalisation context.

## Common input contract

All models receive complete 10-second windows at 125 Hz: 1,250 samples by the
required number of leads. The public loader accepts `.npy`, `.npz`, numeric
`.csv`, and WFDB records and converts them to samples-by-leads before
segmentation.

## Output schemas

Binary models return:

| Column | Meaning |
|---|---|
| `raw_class` | `acceptable` or `unacceptable` |
| `display_class` | `usable` or `unusable` |
| `prob_unacceptable` | Poor-quality probability |
| `prob_acceptable` | Acceptable-quality probability |

Three-class models return `raw_class`, `display_class`, `prob_good`,
`prob_medium`, and `prob_bad`.

The record runner adds record identifiers, segment indices, start/end time,
model name, and input path. See [Inference and Docker](inference.md).

## Intended use

These are research artifacts for reproducible ECG quality experiments. They
are not medical devices and must not be treated as independent evidence for
clinical diagnosis or deployment readiness.
