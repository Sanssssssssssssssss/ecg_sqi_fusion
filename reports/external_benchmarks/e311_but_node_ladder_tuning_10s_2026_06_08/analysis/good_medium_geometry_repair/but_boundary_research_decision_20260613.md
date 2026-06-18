# BUT Boundary Research Decision - 2026-06-13

Selection still uses Clean/SemiClean/node diagnostic only. Original BUT remains bucketed report-only.

## Current State

- Clean frontier is solved at N7200: acc `0.995185`, macro-F1 `0.994917`, good/medium/bad recall `0.999167/0.995972/0.986778`.
- Best balanced original-test diagnostic is the simple morphology/SQI `axis` mode: acc `0.908104`, good/medium/bad recall `0.908242/0.933800/0.630170`.
- More aggressive `axis2` reaches original-test acc `0.910228` and bad recall `0.671533`, but hurts original-all generalization more, so treat it as stress-search rather than the main result.

## Main Breakthrough

The original test set now separates into learnable body versus conflict/stress slices:

| diagnostic subset | n | coverage | axis acc | good | medium | bad |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| high_conf_plus_badcore | 3295 | 0.389 | 0.993020 | 0.997290 | 0.988824 | 1.000000 |
| high_soft_plus_badcore | 5254 | 0.620 | 0.985915 | 0.986232 | 0.985115 | 1.000000 |
| in_domain_all_plus_badcore | 6858 | 0.809 | 0.961942 | 0.978770 | 0.949339 | 1.000000 |
| conflict_only | 1327 | 0.157 | 0.724190 | 0.696035 | 0.785203 | n/a |
| bad_stress_only | 292 | 0.034 | 0.479452 | n/a | n/a | 0.479452 |
| full_original_test | 8477 | 1.000 | 0.908104 | 0.908242 | 0.933800 | 0.630170 |

Interpretation: the learnable main body is already near or above the target. The remaining gap is concentrated in low-margin good/medium conflict and bad outlier stress.

## What Did Not Work

- Ordinary N7200 bad-stress expansion did not promote. Controlled bad was safe-ish but did not improve; high-frequency/core bad made the model too medium-heavy.
- Direct boundary block additions around N7110/N7125 did not beat the previous best ordinary checkpoint.
- QRS-low single-checkpoint conversion failed; the transparent rule works, but synthetic rows alone created unstable class bias.
- Original train/val feature-only classifiers do not solve original test. Best record-normalized probe is only acc `0.842633`, with bad recall `0.048662`.
- A single broad ECG quality axis is not enough for good/medium. The best simple axis for good-vs-medium reaches only balanced acc `0.728297`; bad-vs-nonbad is much easier with `quality_minus_artifact` AUC `0.947102`.

## What This Means

This is no longer mainly a PTB capacity problem. Clean/SemiClean is essentially solved. Original BUT contains record/domain and label-geometry conflicts:

- `125001` label-good rows are 100% closer to original train/val medium geometry than good geometry.
- `111001` bad outliers mostly look closer to original train/val medium/good than bad.
- `111001` good/medium low-margin rows are the remaining large learnable frontier.

## Next Research Direction

Do not keep adding tiny threshold patches. Use three broad tracks:

1. **Main trainable track:** generate larger low-margin good/medium blocks that target the in-domain low-margin body, not high-margin label conflicts.
2. **Bad stress track:** keep bad core in clean selection, and treat bad outlier stress as a separately reported stress bucket unless a controlled block proves it does not collide with clean medium.
3. **Domain/label track:** report high-confidence versus ambiguous/conflict subsets explicitly. Any method that tries to force high-margin label-geometry conflicts into the neural checkpoint must be treated as domain adaptation, not ordinary clean promotion.

Core simple-method takeaway: use a two-axis good/medium story, not many micro rules. The useful axes are (1) QRS reliability / template consistency and (2) non-QRS artifact/detail plus baseline/contact. Bad stress can be handled mostly by quality-minus-artifact, but it collides with clean medium if expanded too aggressively.

## Key Artifacts

- `original_learnability_partition_report.md`
- `original_domain_shift_audit_report.md`
- `original_trainval_model_probe_report.md`
- `original_record_normalized_probe_report.md`
- `simple_ecg_axis_probe_report.md`
- `original_but_breakthrough_summary_20260613.md`
- `badstress_repair_report.md`
- `n7188_bad_domain_conflict_report.md`
