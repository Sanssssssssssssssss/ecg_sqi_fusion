# BUT Domain Gap Audit

- Initial synthetic-vs-BUT distance: `0.7638`
- BUT target labels are expert consensus `1/2/3 -> good/medium/bad`.
- Key hypothesis: BUT bad/medium are more about QRS reliability, low amplitude/contact loss, and wearable drift than our original SNR/morph thresholds.

## Median Feature Snapshot

| class | BUT rms | PTB rms | BUT qrs_prom | PTB qrs_prom | BUT drift | PTB drift |
|---|---:|---:|---:|---:|---:|---:|
| good | 0.2878 | 0.1372 | 11.068 | 5.601 | 0.0672 | 0.0423 |
| medium | 0.2211 | 0.1489 | 5.965 | 4.986 | 0.0412 | 0.0628 |
| bad | 0.1554 | 0.2020 | 2.153 | 6.833 | 0.0003 | 0.1013 |
