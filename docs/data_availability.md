# Data Availability

Raw ECG databases are not redistributed. Pipelines obtain the following public
data when required:

| Dataset | Use |
|---|---|
| PhysioNet/CinC Challenge 2011 Set-A | Classical SQI and 12-lead experiments |
| MIT-BIH Noise Stress Test Database | EM/MA noise controls |
| PTB-XL | Clean ECG carriers for gap-fill proposals |
| BUT QDB | Single-lead Transformer experiment |

Generated arrays, tables, figures, predictions, and local raw data remain
outside version control. Only the four inference model artifacts and their
runtime profiles are tracked.

Exact frozen v116 support-pool replay requires its archived support assets.
Without them, the public-data rebuild is a smoke check and records
`historical_support_exact=false`.

