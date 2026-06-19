# SQI exploratory diagnostics

These figures are implementation-check diagnostics for the paper-aligned SQI reproduction line. They are not the final report figure set.

| Figure | What it checks | Report use |
|---|---|---|
| `diag_01_split_balance.png` | Set-a-only class balance, synthetic noisy-poor counts, and source-record leakage check. | Strong report candidate for methods/QC appendix. |
| `diag_02_qrs_detector_counts.png` | wqrs/eplimited count plausibility, detector agreement, and bSQI lead-level behavior. | Good appendix candidate when discussing QRS alignment risk. |
| `diag_03_feature_integrity.png` | 84-feature nonfinite/range/scale checks. | Good methods appendix candidate. |
| `diag_04_pooled_sqi_distributions.png` | Pooled feature distributions by label for all seven SQIs. | Useful appendix candidate; can support explanations of fSQI/basSQI gaps. |
| `diag_05_lead_median_profiles.png` | Median SQI values by lead and label. | Useful for appendix or internal validation. |
| `diag_06_normalization_train_test.png` | Train/test distribution check before and after normalization for lead-II sSQI and kSQI. | Internal QC; mention only if normalization needs defense. |
| `diag_07_noise_allocation_snr.png` | NSTDB segment allocation, em/ma balance, and example per-lead SNR checks. | Good methods/QC appendix candidate. |
| `diag_08_resampling_wave_psd.png` | Raw 500 Hz versus resampled 125 Hz waveform and PSD comparison. | Internal QC; useful if preprocessing details are challenged. |
| `diag_09_qrs_overlay_12lead.png` | 12-lead ECG with cached wqrs and EP Limited detections. | Good appendix candidate for detector validation. |
| `diag_10_raw_seta_qc.png` | Raw Set-a 500 Hz waveform, per-lead amplitude spread, and PSD sanity check. | Good preprocessing appendix candidate. |
| `diag_11_resample_alias_reference.png` | Current 125 Hz cache versus steep reference resampling error plots. | Strong preprocessing/QC appendix candidate. |
| `diag_12_noise_wave12_gallery.png` | Full 12-lead paper-aligned em/ma synthetic noise residual gallery. | Strong augmentation appendix candidate. |
| `diag_13_single_feature_auc_top20.png` | Top-20 single-feature SQI separability across the current paper-aligned 84-feature table. | Good analysis figure for feature behavior. |
| `diag_14_qrs_hr_by_lead.png` | HR-by-lead boxplots and pooled HR distributions from the current QRS cache. | Good detector sanity appendix candidate. |
| `diag_lead_III_7sqi_jitter.png` | III lead profile across all seven SQIs. | Exploratory diagnostic; useful for lead-specific implementation checks. |
| `diag_lead_II_7sqi_jitter.png` | II lead profile across all seven SQIs. | Exploratory diagnostic; useful for lead-specific implementation checks. |
| `diag_lead_I_7sqi_jitter.png` | I lead profile across all seven SQIs. | Exploratory diagnostic; useful for lead-specific implementation checks. |
| `diag_lead_V1_7sqi_jitter.png` | V1 lead profile across all seven SQIs. | Exploratory diagnostic; useful for lead-specific implementation checks. |
| `diag_lead_V2_7sqi_jitter.png` | V2 lead profile across all seven SQIs. | Exploratory diagnostic; useful for lead-specific implementation checks. |
| `diag_lead_V3_7sqi_jitter.png` | V3 lead profile across all seven SQIs. | Exploratory diagnostic; useful for lead-specific implementation checks. |
| `diag_lead_V4_7sqi_jitter.png` | V4 lead profile across all seven SQIs. | Exploratory diagnostic; useful for lead-specific implementation checks. |
| `diag_lead_V5_7sqi_jitter.png` | V5 lead profile across all seven SQIs. | Exploratory diagnostic; useful for lead-specific implementation checks. |
| `diag_lead_V6_7sqi_jitter.png` | V6 lead profile across all seven SQIs. | Exploratory diagnostic; useful for lead-specific implementation checks. |
| `diag_lead_aVF_7sqi_jitter.png` | aVF lead profile across all seven SQIs. | Exploratory diagnostic; useful for lead-specific implementation checks. |
| `diag_lead_aVL_7sqi_jitter.png` | aVL lead profile across all seven SQIs. | Exploratory diagnostic; useful for lead-specific implementation checks. |
| `diag_lead_aVR_7sqi_jitter.png` | aVR lead profile across all seven SQIs. | Exploratory diagnostic; useful for lead-specific implementation checks. |
| `diag_sqi_bSQI_12lead_jitter.png` | bSQI jitter distribution by class across all 12 leads. | Exploratory diagnostic; use only if a specific SQI behavior needs illustration. |
| `diag_sqi_basSQI_12lead_jitter.png` | basSQI jitter distribution by class across all 12 leads. | Exploratory diagnostic; use only if a specific SQI behavior needs illustration. |
| `diag_sqi_fSQI_12lead_jitter.png` | fSQI jitter distribution by class across all 12 leads. | Exploratory diagnostic; use only if a specific SQI behavior needs illustration. |
| `diag_sqi_iSQI_12lead_jitter.png` | iSQI jitter distribution by class across all 12 leads. | Exploratory diagnostic; use only if a specific SQI behavior needs illustration. |
| `diag_sqi_kSQI_12lead_jitter.png` | kSQI jitter distribution by class across all 12 leads. | Exploratory diagnostic; use only if a specific SQI behavior needs illustration. |
| `diag_sqi_pSQI_12lead_jitter.png` | pSQI jitter distribution by class across all 12 leads. | Exploratory diagnostic; use only if a specific SQI behavior needs illustration. |
| `diag_sqi_sSQI_12lead_jitter.png` | sSQI jitter distribution by class across all 12 leads. | Exploratory diagnostic; use only if a specific SQI behavior needs illustration. |

Contact sheet:

- `diagnostic_contact_sheet.png`
- `diagnostic_contact_sheet.pdf`
