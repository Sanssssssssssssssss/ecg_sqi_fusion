# E3.11f PTB SQI Three-Class Controls

This is a standalone control suite. It uses traditional SQI features on the current E3.11f PTB artifact and does not modify `src/sqi_pipeline`.

Task: `good / medium / bad` three-class compatibility control.

## Data

- source artifact: `E:\GPTProject2\ecg\outputs\experiment\e311_morph_denoise_gap5_7_grid\data\med6p25_badgap7_badcm0p75`
- split counts: `{'test': 2202, 'train': 10935, 'val': 2184}`
- class counts: `{'bad': 5107, 'good': 5107, 'medium': 5107}`
- feature columns: `['I__iSQI', 'I__bSQI', 'I__pSQI', 'I__sSQI', 'I__kSQI', 'I__fSQI', 'I__basSQI']`
- label columns not used as features: `['y']`
- forbidden feature columns detected: `[]`

## Results

| Model | Features | Test Acc | Balanced Acc | Macro F1 | Good R | Medium R | Bad R |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SQI SVM-RBF | 7 SQI | 0.7243 | 0.7243 | 0.7250 | 0.6240 | 0.7466 | 0.8025 |
| SQI MLP | 7 SQI | 0.7030 | 0.7030 | 0.7027 | 0.6294 | 0.6608 | 0.8188 |
| Uformer mainline | waveform + detached Uformer tokens | 0.9882 | - | - | 0.9891 | 0.9796 | 0.9959 |

## Interpretation

- These are traditional SQI-feature three-class compatibility controls, not the original binary SQI paper task.
- They are intentionally kept separate from the mainline and from previous experiment lineage.
- Poor or imbalanced performance is still useful evidence: it shows what handcrafted SQI features can and cannot explain on this synthetic PTB three-class task.

## Artifacts

- normalized summary: `E:\GPTProject2\ecg\outputs\controls\e311f_ptb_sqi_three_class\summary.json`
- raw adapter summary: `E:/GPTProject2/ecg/outputs/controls/e311f_ptb_sqi_three_class/three_class_summary.json`
- per-model reports and predictions live under `outputs/controls/e311f_ptb_sqi_three_class/models/`.
