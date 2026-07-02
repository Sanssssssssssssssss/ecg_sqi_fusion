# SQI supplemental protocol experiments

## Strict Table 6 subset selection

- Paper selected five: `bSQI,basSQI,kSQI,sSQI,fSQI`.
- Validation-recovered five: `bSQI,iSQI,pSQI,fSQI,basSQI`.
- Same set as paper: `False`.
- Protocol: all 127 non-empty SQI subsets are selected by validation accuracy before test evaluation.

| cardinality | Selected_SQI                         | val_Ac   | test_Ac  | test_AUC |
| ----------- | ------------------------------------ | -------- | -------- | -------- |
| 1           | kSQI                                 | 0.943966 | 0.91342  | 0.952024 |
| 2           | kSQI,fSQI                            | 0.961207 | 0.917749 | 0.957121 |
| 3           | bSQI,fSQI,basSQI                     | 0.956897 | 0.935065 | 0.98006  |
| 4           | bSQI,iSQI,fSQI,basSQI                | 0.961207 | 0.935065 | 0.986432 |
| 5           | bSQI,iSQI,pSQI,fSQI,basSQI           | 0.961207 | 0.935065 | 0.983208 |
| 6           | bSQI,iSQI,kSQI,sSQI,fSQI,basSQI      | 0.956897 | 0.939394 | 0.982084 |
| 7           | bSQI,iSQI,kSQI,sSQI,pSQI,fSQI,basSQI | 0.956897 | 0.939394 | 0.980885 |

## Selected-five model diagnostics

| model             | metric | estimate | ci_low   | ci_high  | n_bootstrap_valid | threshold |
| ----------------- | ------ | -------- | -------- | -------- | ----------------- | --------- |
| MLP selected-five | Ac     | 0.917749 | 0.879066 | 0.95279  | 2000              | 0.6295    |
| MLP selected-five | Se     | 0.931034 | 0.882883 | 0.974795 | 2000              | 0.6295    |
| MLP selected-five | Sp     | 0.904348 | 0.842105 | 0.953271 | 2000              | 0.6295    |
| MLP selected-five | AUC    | 0.955247 | 0.918601 | 0.984619 | 2000              | 0.6295    |
| SVM selected-five | Ac     | 0.939394 | 0.904105 | 0.967078 | 2000              | 0.5215    |
| SVM selected-five | Se     | 0.965517 | 0.927262 | 0.991668 | 2000              | 0.5215    |
| SVM selected-five | Sp     | 0.913043 | 0.852933 | 0.960938 | 2000              | 0.5215    |
| SVM selected-five | AUC    | 0.961544 | 0.93056  | 0.985102 | 2000              | 0.5215    |

## fSQI mechanism

| sample_group          | n   | median_log10_abs_diff | q25_log10_abs_diff | q75_log10_abs_diff | flat_fraction_1e-4_median |
| --------------------- | --- | --------------------- | ------------------ | ------------------ | ------------------------- |
| original acceptable   | 773 | -1.93199              | -2.02199           | -1.81488           | 0.005004                  |
| original unacceptable | 225 | -2.0613               | -12                | -1.80616           | 0.125234                  |
| synthetic em          | 274 | -1.45091              | -1.55326           | -1.36345           | 0.00173472                |
| synthetic ma          | 274 | -1.09591              | -1.25608           | -0.942704          | 0.000800641               |

## Cross-noise generalization

| scenario                        | test_Ac  | test_Se  | test_Sp   | test_AUC |
| ------------------------------- | -------- | -------- | --------- | -------- |
| train_em_test_ma                | 0.980892 | 1        | 0.926829  | 0.999369 |
| train_ma_test_em                | 0.853503 | 1        | 0.439024  | 0.99979  |
| synthetic_poor_to_original_poor | 0.791946 | 1        | 0.0606061 | 0.428683 |
| original_poor_to_synthetic_poor | 0.757576 | 0.974138 | 0.45122   | 0.959209 |

## Remaining Evidence for Final Claims

- Absolute paper accuracy is not strictly reproducible because the expert-adjudicated paper labels are unavailable.
- SQI fusion remains effective, but the exact five-SQI optimum is assessed as a validation-selected subset rather than assumed from the paper.
- Synthetic poor and original poor are treated as separate mechanisms when interpreting aggregate performance.
- fSQI is reported as a feature-level mechanism check, not only as a scalar distribution.

### Diagnostic analyses

- Provenance groups were derived after merging `record84_norm.parquet` with the paper-balanced split table. Records were assigned to `original acceptable`, `original unacceptable`, `synthetic em`, or `synthetic ma` from `y`, `is_augmented`, and `noise_type`; `source_record_id` was retained to link each noisy derivative to its clean source.
- PCA used the 84 normalized SQI features (`12 leads x 7 SQIs`). Standardization and the two-component PCA were fitted on original Set-a records only (`original acceptable` plus `original unacceptable`), then applied unchanged to all paper-aligned records including `paper EM` and `paper MA`.
- The domain classifier used poor records only, with original unacceptable coded as 0 and synthetic em/ma as 1. A standardized logistic regression was evaluated by `StratifiedGroupKFold`, grouped by `source_record_id`, and summarized by AUC.
- Distribution shift was also tested by RBF-MMD on standardized poor-record SQI features. The RBF bandwidth used the median-distance heuristic, and significance used a permutation null with the reported number of permutations.
- Cross-domain transfer used the selected-five SQI RBF-SVM (`C=1`, `gamma=0.14`). Each run trained and validated on original acceptable plus one poor-domain source, selected the operating threshold on validation accuracy, refit on train+validation, and evaluated once on original acceptable plus the target poor domain.
- Subgroup AUC/recall analyses used the same train/validation/test split and validation-selected thresholds. Acceptable records report acceptable specificity; original unacceptable, synthetic em, and synthetic ma report poor recall.
- fSQI threshold sweep recomputed 12 lead-specific fSQI values from 125 Hz waveforms as the fraction of adjacent absolute differences below each flatness threshold, then evaluated fixed-RBF models under the same validation-threshold protocol.
- basSQI paired deltas used `1-basSQI = P_0-1 / P_0-40` and compared each synthetic noisy record with its matched clean `source_record_id`, isolating the augmentation-induced change in low-frequency power fraction.

### Paper quintuplet rank

| paper_selected_five        | paper_subset_id_in_strict_table | validation_rank_among_21_five_sqi_subsets | n_five_sqi_subsets | paper_val_Ac | paper_val_AUC | best_five_subset           | best_five_val_Ac | best_five_val_AUC | paper_gap_to_best_val_Ac | paper_gap_to_best_val_Ac_pp | one_validation_error | one_validation_error_pp | within_one_validation_error_of_best |
| -------------------------- | ------------------------------- | ----------------------------------------- | ------------------ | ------------ | ------------- | -------------------------- | ---------------- | ----------------- | ------------------------ | --------------------------- | -------------------- | ----------------------- | ----------------------------------- |
| bSQI,basSQI,kSQI,sSQI,fSQI | bSQI+kSQI+sSQI+fSQI+basSQI      | 6                                         | 21                 | 0.952586     | 0.988109      | bSQI,iSQI,pSQI,fSQI,basSQI | 0.961207         | 0.980158          | 0.00862069               | 0.862069                    | 0.00431034           | 0.431034                | False                               |

### Five-SQI validation plateau

| cardinality | n_validation_records | best_val_Ac | one_validation_error | n_subsets_within_one_error | fraction_subsets_within_one_error | paper_subset_rank | paper_gap_to_best_val_Ac | interpretation              |
| ----------- | -------------------- | ----------- | -------------------- | -------------------------- | --------------------------------- | ----------------- | ------------------------ | --------------------------- |
| 5           | 232                  | 0.961207    | 0.00431034           | 5                          | 0.238095                          | 6                 | 0.00862069               | changed_selection_structure |

### SQI domain shift

| metric                             | estimate | comparison                      | bootstrap_or_test                        | n_original_poor | n_synthetic_poor | p_value_permutation | n_permutations | kernel_gamma_median_heuristic |
| ---------------------------------- | -------- | ------------------------------- | ---------------------------------------- | --------------- | ---------------- | ------------------- | -------------- | ----------------------------- |
| source_grouped_logistic_domain_auc | 0.973536 | original poor vs synthetic poor | StratifiedGroupKFold by source_record_id | 225             | 548              |                     |                |                               |
| RBF-MMD2                           | 0.330871 | original poor vs synthetic poor |                                          | 225             | 548              | 0.000999001         | 1000           | 0.00640043                    |

### Cross-domain AUC matrix source table

| train_poor_domain | test_poor_domain | test_Ac  | test_Se  | test_Sp   | test_AUC |
| ----------------- | ---------------- | -------- | -------- | --------- | -------- |
| original poor     | original poor    | 0.899329 | 0.974138 | 0.636364  | 0.911964 |
| original poor     | em               | 0.738854 | 0.974138 | 0.0731707 | 0.933137 |
| original poor     | ma               | 0.936306 | 0.974138 | 0.829268  | 0.985282 |
| em                | original poor    | 0.791946 | 1        | 0.0606061 | 0.431296 |
| em                | em               | 1        | 1        | 1         | 1        |
| em                | ma               | 0.980892 | 1        | 0.926829  | 0.999369 |
| ma                | original poor    | 0.778523 | 1        | 0         | 0.432341 |
| ma                | em               | 0.853503 | 1        | 0.439024  | 0.99979  |
| ma                | ma               | 0.987261 | 1        | 0.95122   | 1        |
| synthetic poor    | original poor    | 0.791946 | 1        | 0.0606061 | 0.428683 |
| synthetic poor    | em               | 1        | 1        | 1         | 1        |
| synthetic poor    | ma               | 1        | 1        | 1         | 1        |

### fSQI fixed-RBF threshold scan

| threshold_mv | val_Ac   | test_Ac  | test_Se  | test_Sp  | test_AUC |
| ------------ | -------- | -------- | -------- | -------- | -------- |
| 1e-06        | 0.616379 | 0.580087 | 1        | 0.156522 | 0.465855 |
| 3e-06        | 0.616379 | 0.580087 | 1        | 0.156522 | 0.39018  |
| 1e-05        | 0.616379 | 0.580087 | 1        | 0.156522 | 0.314055 |
| 3e-05        | 0.616379 | 0.580087 | 1        | 0.156522 | 0.262519 |
| 0.0001       | 0.616379 | 0.580087 | 1        | 0.156522 | 0.248951 |
| 0.0003       | 0.857759 | 0.774892 | 0.655172 | 0.895652 | 0.881484 |
| 0.001        | 0.866379 | 0.891775 | 0.982759 | 0.8      | 0.911619 |
| 0.003        | 0.866379 | 0.891775 | 0.922414 | 0.86087  | 0.926087 |
| 0.01         | 0.887931 | 0.909091 | 0.965517 | 0.852174 | 0.940105 |

### fSQI linear vs RBF

| model      | threshold_mv | test_Ac  | test_Se  | test_Sp  | test_AUC |
| ---------- | ------------ | -------- | -------- | -------- | -------- |
| logistic   | 0.0001       | 0.580087 | 1        | 0.156522 | 0.363118 |
| linear-SVM | 0.0001       | 0.636364 | 0.362069 | 0.913043 | 0.626312 |
| RBF-SVM    | 0.0001       | 0.580087 | 1        | 0.156522 | 0.248951 |

### basSQI mechanism

`1-basSQI = P_0-1 / P_0-40`, the low-frequency power fraction.

| model                | sample_group          | n   | metric                 | value    | threshold | score_median |
| -------------------- | --------------------- | --- | ---------------------- | -------- | --------- | ------------ |
| basSQI fixed RBF-SVM | original acceptable   | 116 | acceptable_specificity | 0.87069  | 0.594     | 0.903763     |
| basSQI fixed RBF-SVM | original unacceptable | 33  | poor_recall            | 0.666667 | 0.594     | 0.222587     |
| basSQI fixed RBF-SVM | synthetic em          | 41  | poor_recall            | 1        | 0.594     | 0.120999     |
| basSQI fixed RBF-SVM | synthetic ma          | 41  | poor_recall            | 0.97561  | 0.594     | 0.0380878    |

### SQI subgroup separability

| SQI    | poor_domain   | test_Ac  | test_Se  | test_Sp  | test_AUC |
| ------ | ------------- | -------- | -------- | -------- | -------- |
| bSQI   | original poor | 0.885906 | 0.982759 | 0.545455 | 0.868861 |
| bSQI   | em            | 0.898089 | 0.982759 | 0.658537 | 0.95963  |
| bSQI   | ma            | 0.968153 | 1        | 0.878049 | 0.987805 |
| basSQI | original poor | 0.885906 | 0.991379 | 0.515152 | 0.82837  |
| basSQI | em            | 0.980892 | 0.974138 | 1        | 0.988856 |
| basSQI | ma            | 0.961783 | 0.974138 | 0.926829 | 0.988646 |
| kSQI   | original poor | 0.872483 | 1        | 0.424242 | 0.869906 |
| kSQI   | em            | 0.987261 | 0.982759 | 1        | 0.997687 |
| kSQI   | ma            | 0.987261 | 0.982759 | 1        | 0.99979  |
| sSQI   | original poor | 0.865772 | 0.948276 | 0.575758 | 0.86442  |
| sSQI   | em            | 0.993631 | 0.991379 | 1        | 0.997687 |
| sSQI   | ma            | 0.980892 | 0.991379 | 0.95122  | 0.999579 |
| fSQI   | original poor | 0.899329 | 1        | 0.545455 | 0.740596 |
| fSQI   | em            | 0.936306 | 0.939655 | 0.926829 | 0.976451 |
| fSQI   | ma            | 0.961783 | 0.982759 | 0.902439 | 0.993902 |
| iSQI   | original poor | 0.845638 | 0.87931  | 0.727273 | 0.829415 |
| iSQI   | em            | 0.910828 | 0.948276 | 0.804878 | 0.957107 |
| iSQI   | ma            | 0.917197 | 0.965517 | 0.780488 | 0.949117 |
| pSQI   | original poor | 0.852349 | 1        | 0.333333 | 0.668887 |
| pSQI   | em            | 0.738854 | 0.956897 | 0.121951 | 0.87931  |
| pSQI   | ma            | 0.910828 | 0.991379 | 0.682927 | 0.923886 |

### Subset subgroup recall

| model            | sample_group          | n   | metric                 | value    | threshold | score_median |
| ---------------- | --------------------- | --- | ---------------------- | -------- | --------- | ------------ |
| iSQI             | original acceptable   | 116 | acceptable_specificity | 0.715517 | 0.606     | 0.941465     |
| iSQI             | original unacceptable | 33  | poor_recall            | 0.909091 | 0.606     | 0.241457     |
| iSQI             | synthetic em          | 41  | poor_recall            | 0.878049 | 0.606     | 0.218783     |
| iSQI             | synthetic ma          | 41  | poor_recall            | 0.926829 | 0.606     | 0.18557      |
| basSQI           | original acceptable   | 116 | acceptable_specificity | 0.87069  | 0.594     | 0.903763     |
| basSQI           | original unacceptable | 33  | poor_recall            | 0.666667 | 0.594     | 0.222587     |
| basSQI           | synthetic em          | 41  | poor_recall            | 1        | 0.594     | 0.120999     |
| basSQI           | synthetic ma          | 41  | poor_recall            | 0.97561  | 0.594     | 0.0380878    |
| paper pair       | original acceptable   | 116 | acceptable_specificity | 0.965517 | 0.3105    | 0.953307     |
| paper pair       | original unacceptable | 33  | poor_recall            | 0.545455 | 0.3105    | 0.263146     |
| paper pair       | synthetic em          | 41  | poor_recall            | 0.926829 | 0.3105    | 0.0444503    |
| paper pair       | synthetic ma          | 41  | poor_recall            | 0.97561  | 0.3105    | 0.0111494    |
| paper quintuplet | original acceptable   | 116 | acceptable_specificity | 0.982759 | 0.3745    | 0.976298     |
| paper quintuplet | original unacceptable | 33  | poor_recall            | 0.69697  | 0.3745    | 0.0761671    |
| paper quintuplet | synthetic em          | 41  | poor_recall            | 1        | 0.3745    | 0.029619     |
| paper quintuplet | synthetic ma          | 41  | poor_recall            | 1        | 0.3745    | 0.00590194   |
| all seven        | original acceptable   | 116 | acceptable_specificity | 0.965517 | 0.471     | 0.977881     |
| all seven        | original unacceptable | 33  | poor_recall            | 0.69697  | 0.471     | 0.0858721    |
| all seven        | synthetic em          | 41  | poor_recall            | 1        | 0.471     | 0.0196606    |
| all seven        | synthetic ma          | 41  | poor_recall            | 1        | 0.471     | 0.00483615   |

### Pair-to-quintuplet error rescue

| sample_group          | pair_wrong_n | pair_wrong_quintuplet_correct_n | rescue_rate | pair_error_rate | quintuplet_error_rate |
| --------------------- | ------------ | ------------------------------- | ----------- | --------------- | --------------------- |
| original acceptable   | 4            | 3                               | 0.75        | 0.0344828       | 0.0172414             |
| original unacceptable | 15           | 5                               | 0.333333    | 0.454545        | 0.30303               |
| synthetic em          | 3            | 3                               | 1           | 0.0731707       | 0                     |
| synthetic ma          | 1            | 1                               | 1           | 0.0243902       | 0                     |

## Figure index

- `fig_supp_01_strict_table6_subset_selection`: strict 127-subset validation selection and inclusion heatmap.
- `fig_supp_02_model_stratified_diagnostics`: selected-five ROC, score distributions, and source-bootstrap CI.
- `fig_supp_03_fsqi_mechanism`: log-difference distributions and flat-threshold sensitivity.
- `fig_12_fsqi_mechanism`: updated fSQI lead-level mechanism, fixed-RBF threshold scan, and subgroup recall.
- `fig_13_sqi_domain_shift`: PCA, per-SQI domain AUC, and cross-domain AUC matrix.
- `fig_14_bassqi_domain_shift`: conditional basSQI mechanism figure when basSQI shows strong domain shift.
- `fig_15_sqi_subgroup_separability`: SQI by poor-domain AUC, subset subgroup recall, and pair-to-quintuplet error rescue.
- `model_diagnostics/error_gallery/high_confidence_gallery/*`: high-confidence error/control waveform review pack.

Shared image copies:

- `fig_10_strict_table6_subset_selection.png`
- `fig_10_strict_table6_subset_selection.pdf`
- `fig_10_strict_table6_subset_selection.svg`
- `fig_11_model_stratified_diagnostics.png`
- `fig_11_model_stratified_diagnostics.pdf`
- `fig_11_model_stratified_diagnostics.svg`
- `fig_12_fsqi_mechanism.png`
- `fig_12_fsqi_mechanism.pdf`
- `fig_12_fsqi_mechanism.svg`
- `fig_12_fsqi_mechanism.png`
- `fig_12_fsqi_mechanism.pdf`
- `fig_12_fsqi_mechanism.svg`
- `fig_13_sqi_domain_shift.png`
- `fig_13_sqi_domain_shift.pdf`
- `fig_13_sqi_domain_shift.svg`
- `fig_14_bassqi_domain_shift.png`
- `fig_14_bassqi_domain_shift.pdf`
- `fig_14_bassqi_domain_shift.svg`
- `fig_15_sqi_subgroup_separability.png`
- `fig_15_sqi_subgroup_separability.pdf`
- `fig_15_sqi_subgroup_separability.svg`

## Reproducibility commands

```powershell
.\.venv\Scripts\python.exe -m supplemental_sqi_experiments.run diagnose-existing
.\.venv\Scripts\python.exe -m supplemental_sqi_experiments.run final-claims
.\.venv\Scripts\python.exe -m supplemental_sqi_experiments.run build-isolated --seed 0
.\.venv\Scripts\python.exe -m supplemental_sqi_experiments.run stability
.\.venv\Scripts\python.exe -m supplemental_sqi_experiments.run stability --include-mlp
```
