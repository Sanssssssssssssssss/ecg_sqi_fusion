# N17043 Teacher-Full UFormer Transfer Attempt

## Question

Can we convert the teacher / visual-boundary discoveries into larger PTB synthetic blocks and let an ordinary UFormer checkpoint learn the good/medium separation directly?

## Answer

Not yet. The new full-shell synthetic blocks are learnable on PTB-style synthetic validation, but they do not transfer cleanly to the full N17043 BUT-target shell. The ordinary checkpoint either becomes medium-heavy and loses many good rows, or becomes more balanced but still misses the promotion gate by a large margin.

The teacher / rule models are therefore useful as separability probes and generator design signals, not as a final replacement for UFormer.

## Variants Tested

All variants are ordinary UFormer training attempts from the boundary-block generator path. Original BUT metrics remain report-only and are not used for selection.

| Config | Raw Acc | Raw Good | Raw Medium | Raw Bad | Calibrated Acc | Feature Diagnostic Acc | Original Report Acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| teacherfull_balanced | 0.614832 | 0.319838 | 0.951167 | 0.970617 | 0.623114 | 0.870162 | 0.718650 |
| teacherfull_goodpush | 0.709558 | 0.482016 | 0.974125 | 0.970617 | 0.711604 | 0.870162 | 0.684794 |
| teacherfull_mediumwall | 0.648937 | 0.369066 | 0.974125 | 0.970617 | 0.654920 | 0.870162 | 0.697299 |
| teacherfull_gooddominant | 0.819965 | 0.798040 | 0.796763 | 0.971841 | 0.817761 | 0.870162 | 0.716999 |
| teacherfull_goodwide | 0.773673 | 0.823975 | 0.617332 | 0.970617 | 0.767029 | 0.870162 | 0.615430 |

The feature diagnostic mode shown above is the existing report-only feature rule (`feature_pc1_qrsprom_tree_mediumveto_n12800_trainval`). It is not an ordinary neural checkpoint and does not count as model promotion.

## Interpretation

1. Bad is not the main blocker in this round. Bad recall stays near 0.97-1.00 on node diagnostics.
2. The failure is still good/medium geometry transfer. Medium-heavy synthetic blocks make good collapse; good-heavy blocks recover good but lose medium.
3. The best ordinary full-shell attempt is `teacherfull_gooddominant`, but it only reaches raw `acc=0.819965`, with good/medium recalls both around 0.80.
4. This means the current PTB generator can produce rows that UFormer learns internally, but those rows still do not match the true BUT outer-shell morphology well enough.

## Next Experimental Direction

Do not keep only increasing row counts from the same synthetic recipe. The next useful step is to analyze the `teacherfull_gooddominant` remaining good/medium mistakes at waveform level and add morphology-matched generator controls for the failed shell:

- good outer shell that looks visually good but has low QRS confidence / baseline complexity,
- medium wall that remains QRS-visible but has degraded detail,
- medium hard negatives with low QRS visibility and stronger non-QRS dynamics,
- controlled bad outlier only as a small guardrail block.

The goal remains an ordinary promoted checkpoint. Teacher/rule artifacts should stay as analysis tools unless explicitly promoted as a transparent rule-engine artifact.
