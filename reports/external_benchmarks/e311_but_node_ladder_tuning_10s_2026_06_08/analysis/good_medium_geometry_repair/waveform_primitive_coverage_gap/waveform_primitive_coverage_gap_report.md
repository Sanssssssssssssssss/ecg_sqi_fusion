# Waveform Primitive Coverage Gap

Report-only diagnostic. It compares BUT original-test rows with PTB synthetic train rows in waveform-computable primitive-stat space.

Candidate: `predtop20_sqiquery_subject111_shift_stress_pretrain`
Primitive bank: `qrs_stress_v5`

## Headline

- correct median NN distance: `16.8151`
- wrong median NN distance: `25.8110`
- correct same-class nearest-neighbor rate: `0.939`
- wrong same-class nearest-neighbor rate: `0.319`
- bad-outlier wrong median NN distance: `29.1888`
- medium->good median NN distance: `42.6788`
- good->medium median NN distance: `16.0797`

## By True/Region

true,region,n,err_rate,nn_dist_median,nn_dist_p90,same_class_nn_rate,same_region_nn_rate,pred_top
bad,outlier_low_confidence,292,0.9623287671232876,29.1688175201416,61.25093231201174,0.0,0.0,good
medium,outlier_low_confidence,2179,0.30931620009178523,23.528661727905273,65.10229034423828,0.7627351996328591,0.0,medium
good,outlier_low_confidence,2411,0.1646619659892161,19.383729934692383,26.248903274536133,0.841974284529241,0.0,good
good,good_medium_overlap,1191,0.11922753988245172,16.208415985107422,20.37492561340332,0.8446683459277917,0.0,good
bad,near_bad_boundary,119,0.03361344537815126,26.584842681884766,29.313093185424805,1.0,0.0,bad
medium,good_medium_overlap,1711,0.0011689070718877848,11.725773811340332,16.26604461669922,0.9596727060198714,0.0,medium
medium,clean_core,531,0.0,11.604405403137207,15.070694923400879,0.9943502824858758,0.0,medium
good,clean_core,38,0.0,11.67452621459961,16.033987045288086,0.9473684210526315,0.0,good
medium,medium_bad_overlap,5,0.0,12.958784103393555,15.471744155883789,0.8,0.0,medium


## By Error Type

true,pred_raw,region,n,err_rate,nn_dist_median,nn_dist_p90,same_class_nn_rate,same_region_nn_rate,pred_top
medium,good,outlier_low_confidence,633,1.0,42.67884063720703,92.97588806152346,0.5292259083728278,0.0,good
good,medium,outlier_low_confidence,389,1.0,16.838783264160156,23.90252342224121,0.17480719794344474,0.0,medium
bad,good,outlier_low_confidence,205,1.0,31.04923439025879,64.96090240478512,0.0,0.0,good
good,medium,good_medium_overlap,142,1.0,14.534637451171875,18.1261287689209,0.2746478873239437,0.0,medium
bad,medium,outlier_low_confidence,76,1.0,21.725318908691406,43.52668380737305,0.0,0.0,medium
medium,bad,outlier_low_confidence,41,1.0,74.37751007080078,158.8804931640625,0.6341463414634146,0.0,bad
good,bad,outlier_low_confidence,8,1.0,64.44454956054688,58872.5859375,0.75,0.0,bad
bad,medium,near_bad_boundary,4,1.0,29.495445251464844,30.60621871948242,1.0,0.0,medium
medium,good,good_medium_overlap,2,1.0,19635.994140625,35325.901953125,0.0,0.0,good
good,good,outlier_low_confidence,2014,0.0,19.779727935791016,26.443969154357912,0.971201588877855,0.0,good
medium,medium,good_medium_overlap,1709,0.0,11.72289752960205,16.218995666503908,0.9607957870099474,0.0,medium
medium,medium,outlier_low_confidence,1505,0.0,19.99262237548828,38.1155502319336,0.8644518272425249,0.0,medium
good,good,good_medium_overlap,1049,0.0,16.444242477416992,20.635787200927734,0.9218303145853194,0.0,good
medium,medium,clean_core,531,0.0,11.604405403137207,15.070694923400879,0.9943502824858758,0.0,medium
bad,bad,near_bad_boundary,115,0.0,26.412092208862305,29.227281188964845,1.0,0.0,bad
good,good,clean_core,38,0.0,11.67452621459961,16.033987045288086,0.9473684210526315,0.0,good
bad,bad,outlier_low_confidence,11,0.0,22.78673553466797,73.16448974609375,0.0,0.0,bad
medium,medium,medium_bad_overlap,5,0.0,12.958784103393555,15.471744155883789,0.8,0.0,medium

