# E3.11f BUT Protocol-First Adaptation

This run supersedes the previous 14h BUT grid until the BUT window protocol is audited.

## Stages

- Stage 0: protocol sweep for current 10s, 10s purity, 5s crops, 5s stride, and two-crop ensemble.
- Stage 1: false-negative visual/failure taxonomy.
- Stage 2: generator/head-only plan, gated on Stage 0/1 findings.

## Guardrails

- BUT mapping remains `1/2/3 -> good/medium/bad`.
- Calibration is validation-only; test is never used for threshold selection.
- `src/sqi_pipeline` and mainline checkpoints are not modified.

Output root: `E:\GPTProject2\ecg\outputs\external_benchmarks\e311_but_protocol_adaptation_2026_06_03`
