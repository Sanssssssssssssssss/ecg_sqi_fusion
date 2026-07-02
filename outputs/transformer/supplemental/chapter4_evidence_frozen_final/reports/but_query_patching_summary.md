# BUT Query Activation Patching

- Rescue pairs: `124`
- Self-patch max logit error: `0`
- Verdict: **supports a GM_BOUNDARY-mediated medium decision path**.
- Interpretation boundary: this is a causal head-path intervention. E31 computes the medium logit from the GM_BOUNDARY representation, so final-state patching proves mediation through that path; probe AUCs below show other query embeddings can still carry class information.
- Patch-location readout: GM_BOUNDARY mean delta is `5.6300` at `Z_Q` and `5.7025` at `Q_e`, so this run does not isolate high-resolution cross-attention as the sole source of the boundary evidence.

## Final-query forward patch

| query | n | mean_delta_h_int | sem_delta_h_int | flip_rate | mean_delta_bad_logit |
| --- | --- | --- | --- | --- | --- |
| GM_BOUNDARY | 124 | 5.7025 | 0.2718 | 0.8226 | 0.0000 |
| BAD_STRESS | 124 | 0.0000 | 0.0000 | 0.0000 | 0.0458 |
| BASELINE | 124 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| CONTACT_RESET | 124 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| DETAIL_NOISE | 124 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| GLOBAL_MORPH | 124 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| QRS | 124 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| RR_TEMPLATE | 124 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Query probe AUC

| query | task | auc | train_n | test_n |
| --- | --- | --- | --- | --- |
| QRS | good_vs_medium | 0.9851 | 13509 | 1685 |
| QRS | bad_vs_nonbad | 0.9998 | 14823 | 1849 |
| RR_TEMPLATE | good_vs_medium | 0.9861 | 13509 | 1685 |
| RR_TEMPLATE | bad_vs_nonbad | 0.9998 | 14823 | 1849 |
| BASELINE | good_vs_medium | 0.9868 | 13509 | 1685 |
| BASELINE | bad_vs_nonbad | 0.9999 | 14823 | 1849 |
| CONTACT_RESET | good_vs_medium | 0.9859 | 13509 | 1685 |
| CONTACT_RESET | bad_vs_nonbad | 0.9999 | 14823 | 1849 |
| DETAIL_NOISE | good_vs_medium | 0.9864 | 13509 | 1685 |
| DETAIL_NOISE | bad_vs_nonbad | 0.9997 | 14823 | 1849 |
| GLOBAL_MORPH | good_vs_medium | 0.9862 | 13509 | 1685 |
| GLOBAL_MORPH | bad_vs_nonbad | 0.9997 | 14823 | 1849 |
| GM_BOUNDARY | good_vs_medium | 0.9855 | 13509 | 1685 |
| GM_BOUNDARY | bad_vs_nonbad | 0.9998 | 14823 | 1849 |
| BAD_STRESS | good_vs_medium | 0.9855 | 13509 | 1685 |
| BAD_STRESS | bad_vs_nonbad | 0.9999 | 14823 | 1849 |
