# Waveform Transformer Ensemble Probe

Report-only probe over waveform-only Transformer candidates. BUT is not used for model selection here; this diagnoses whether candidate errors are complementary.

## Top Original Test Ensembles

| combo | weights | acc | good | medium | bad | nonbad->bad | bad->good | bad->medium |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `predtop20_sqiquery_subject111_impulsebad_dual_p20` | `[1.0]` | 0.822225 | 0.858 | 0.844 | 0.270 | 28 | 190 | 110 |
| `predtop20_sqiquery_subject111_impulsebad_dual_p20+predtop20_eventqrs_impulsebad_dual_p20_qrsheavy` | `[0.5, 0.5]` | 0.821635 | 0.881 | 0.825 | 0.260 | 23 | 201 | 103 |
| `predtop20_sqiquery_subject111_impulsebad_dual_p20+predtop20_eventqrs_impulsebad_dual_p20_qrsheavy+p20_sqiquery_primctx_v5_light` | `[0.5, 0.33, 0.17]` | 0.820927 | 0.879 | 0.824 | 0.277 | 27 | 197 | 100 |
| `predtop20_sqiquery_subject111_impulsebad_dual_p20+predtop20_eventqrs_impulsebad_dual_p20_qrsheavy+p20_sqiquery_primctx_v5_light` | `[0.65, 0.2, 0.15]` | 0.820455 | 0.870 | 0.830 | 0.277 | 27 | 192 | 105 |
| `predtop20_sqiquery_subject111_impulsebad_dual_p20+predtop20_eventqrs_impulsebad_dual_p20_qrsheavy` | `[0.65, 0.35]` | 0.820455 | 0.868 | 0.832 | 0.268 | 25 | 191 | 110 |
| `predtop20_sqiquery_subject111_impulsebad_dual_p20+predtop20_eventqrs_impulsebad_dual_p20_qrsheavy` | `[0.8, 0.2]` | 0.820337 | 0.862 | 0.837 | 0.268 | 27 | 190 | 111 |
| `predtop20_sqiquery_subject111_impulsebad_dual_p20+predtop20_eventqrs_impulsebad_dual_p20_qrsheavy+p20_sqiquery_primctx_v5_light` | `[0.33, 0.5, 0.17]` | 0.820219 | 0.890 | 0.814 | 0.275 | 24 | 210 | 88 |
| `predtop20_sqiquery_subject111_impulsebad_dual_p20+predtop20_eventqrs_impulsebad_dual_p20_qrsheavy+p20_sqiquery_primctx_v5_badguard` | `[0.65, 0.2, 0.15]` | 0.820219 | 0.872 | 0.829 | 0.275 | 26 | 192 | 106 |
| `predtop20_sqiquery_subject111_impulsebad_dual_p20+predtop20_eventqrs_impulsebad_dual_p20_qrsheavy+p20_sqiquery_primctx_v5_light` | `[0.5, 0.2, 0.3]` | 0.819866 | 0.879 | 0.820 | 0.297 | 36 | 198 | 91 |
| `predtop20_sqiquery_subject111_impulsebad_dual_p20+p20_sqiquery_primctx_v5_light` | `[0.8, 0.2]` | 0.819512 | 0.864 | 0.833 | 0.280 | 35 | 192 | 104 |
| `predtop20_sqiquery_subject111_impulsebad_dual_p20+p20_sqiquery_primctx_v5_badguard` | `[0.8, 0.2]` | 0.819394 | 0.866 | 0.831 | 0.280 | 28 | 193 | 103 |
| `predtop20_sqiquery_subject111_impulsebad_dual_p20+predtop20_eventqrs_impulsebad_dual_p20_qrsheavy` | `[0.35, 0.65]` | 0.819276 | 0.890 | 0.818 | 0.209 | 11 | 215 | 110 |
| `predtop20_sqiquery_subject111_impulsebad_dual_p20+predtop20_eventqrs_impulsebad_dual_p20_qrsheavy+p20_sqiquery_primctx_v5_badguard` | `[0.5, 0.33, 0.17]` | 0.819040 | 0.879 | 0.820 | 0.277 | 24 | 198 | 99 |
| `predtop20_sqiquery_subject111_impulsebad_dual_p20+predtop20_eventqrs_impulsebad_dual_p20_qrsheavy+p20_sqiquery_primctx_v5_light` | `[0.2, 0.5, 0.3]` | 0.818922 | 0.894 | 0.806 | 0.290 | 25 | 215 | 77 |
| `predtop20_sqiquery_subject111_impulsebad_dual_p20+predtop20_eventqrs_impulsebad_dual_p20_qrsheavy+p20_sqiquery_primctx_v5_badguard` | `[0.33, 0.5, 0.17]` | 0.818922 | 0.890 | 0.812 | 0.270 | 20 | 212 | 88 |

## Takeaway

Best report-only ensemble original_test acc is `0.822225` with good/medium/bad recalls `0.858/0.844/0.270`.
If this does not exceed the single p20 candidate meaningfully, the tested Transformer variants are not complementary enough; further work should target data/domain coverage rather than simple ensembling.