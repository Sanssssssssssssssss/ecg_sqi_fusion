# BUT Statistics And Visual Analysis Summary

## Morphology Analysis

The morphology-analysis pass extracted comparable features from BUT and synthetic variants:

- QRS reliability: peak prominence, QRS-like density, missing/spurious peak proxies, slope/width.
- Detail reliability: non-QRS energy, local derivative anomalies, P/T/ST instability.
- Wearable/contact signals: flatline/contact spans, clipping, low amplitude, baseline wander, HF bursts.

The distance tables in `analysis/morphology_analysis/` show that feature similarity is useful but not sufficient. Better feature-target scores do not automatically produce better BUT metrics, especially when medium/bad boundaries are wrong.

## Medium Cluster Analysis

The medium analysis showed that BUT class 2 is not simply between class 1 and class 3. It behaves like its own cluster: QRS often remains visible, while detail reliability and local morphology become questionable. This finding is why later generators treat medium as independent instead of a milder bad.

## Visual Evidence

The most important visual examples are:

- `figures/sample_or_medium_confused.png`: many class-2 BUT windows have visible QRS and are predicted too good.
- `figures/sample_or_bad_missed.png`: some class-3 BUT windows keep visible QRS but are globally unreliable.
- `figures/sample_or_subtype_gallery.png`: synthetic bad subtype OR examples; visually plausible but still not enough.

## Current Interpretation

The problem is not "add more noise". The problem is aligning PTB synthetic labels to an expert usability boundary. The next promising direction is diagnostic-usability generation: visible-QRS global bad, independent medium, and strict good.
