# BUT Query Activation Patching

- Rescue pairs: `123`
- Self-patch max logit error: `0`
- Verdict: **supports a distributed boundary decision path, not a GM_BOUNDARY-only claim**.
- Interpretation boundary: this is a causal final-query intervention. The largest Q_e mean delta is `RR_TEMPLATE` at `0.4903`; probe AUCs below only show information content, not causality.
- Patch-location readout: GM_BOUNDARY mean delta is `0.4133` at `Z_Q` and `0.4655` at `Q_e`.

## Final-query forward patch

| query | n | mean_delta_h_int | sem_delta_h_int | flip_rate | mean_delta_bad_logit |
| --- | --- | --- | --- | --- | --- |
| RR_TEMPLATE | 123 | 0.4903 | 0.0274 | 0.0244 | 0.0860 |
| BASELINE | 123 | 0.4889 | 0.0278 | 0.0163 | 0.0806 |
| GLOBAL_MORPH | 123 | 0.4863 | 0.0269 | 0.0163 | 0.0607 |
| CONTACT_RESET | 123 | 0.4813 | 0.0273 | 0.0163 | 0.0687 |
| DETAIL_NOISE | 123 | 0.4813 | 0.0267 | 0.0163 | 0.0687 |
| QRS | 123 | 0.4764 | 0.0258 | 0.0244 | 0.0886 |
| GM_BOUNDARY | 123 | 0.4655 | 0.0253 | 0.0244 | 0.0562 |
| BAD_STRESS | 123 | 0.4367 | 0.0230 | 0.0325 | 0.0433 |

## Query probe AUC

| query | task | auc | train_n | test_n |
| --- | --- | --- | --- | --- |
| QRS | good_vs_medium | 0.9848 | 13509 | 1685 |
| QRS | bad_vs_nonbad | 1.0000 | 14823 | 1849 |
| RR_TEMPLATE | good_vs_medium | 0.9852 | 13509 | 1685 |
| RR_TEMPLATE | bad_vs_nonbad | 1.0000 | 14823 | 1849 |
| BASELINE | good_vs_medium | 0.9854 | 13509 | 1685 |
| BASELINE | bad_vs_nonbad | 1.0000 | 14823 | 1849 |
| CONTACT_RESET | good_vs_medium | 0.9853 | 13509 | 1685 |
| CONTACT_RESET | bad_vs_nonbad | 1.0000 | 14823 | 1849 |
| DETAIL_NOISE | good_vs_medium | 0.9858 | 13509 | 1685 |
| DETAIL_NOISE | bad_vs_nonbad | 1.0000 | 14823 | 1849 |
| GLOBAL_MORPH | good_vs_medium | 0.9859 | 13509 | 1685 |
| GLOBAL_MORPH | bad_vs_nonbad | 1.0000 | 14823 | 1849 |
| GM_BOUNDARY | good_vs_medium | 0.9850 | 13509 | 1685 |
| GM_BOUNDARY | bad_vs_nonbad | 1.0000 | 14823 | 1849 |
| BAD_STRESS | good_vs_medium | 0.9853 | 13509 | 1685 |
| BAD_STRESS | bad_vs_nonbad | 1.0000 | 14823 | 1849 |
