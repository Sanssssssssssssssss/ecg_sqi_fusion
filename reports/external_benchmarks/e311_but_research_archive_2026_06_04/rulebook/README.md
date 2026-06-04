# Synthetic Rulebook

## Base Setup Shared By Generator Experiments

- Source clean ECG: PTB-derived synthetic artifact used by E3.11f, 10s at 125Hz, split 10935/2184/2202.
- Output labels: good/medium/bad three-class SQI aligned to BUT class 1/2/3 semantics.
- Model recipe unless stated otherwise: Uformer1D residual denoiser + detached full-token classifier/SQI head.
- Formal external eval: BUT QDB 10s P1, validation-only threshold or bias calibration, test used only once for reporting.
- PTB sanity checks: internal PTB accuracy, bad recall, and denoise score are retained so external gains do not destroy the mainline.

## Rule Evolution In One Sentence

We moved from "bad means more noise" to "bad means any fatal diagnostic-usability failure", and from "medium is halfway between good and bad" to "medium is QRS usable but detail unreliable".
