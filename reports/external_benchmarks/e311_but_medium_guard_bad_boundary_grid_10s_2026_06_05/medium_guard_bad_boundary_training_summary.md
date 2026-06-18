# Medium-Guard Bad-Boundary Training Summary

Target: original BUT acc >= 0.825, medium >= 0.825, bad >= 0.80. Reference h_bad_rescue_05 macro-F1 0.7454, recalls [0.887, 0.773, 0.793].

| rank | mode | variant | return | orig acc | orig macro | recalls G/M/B | balanced macro | PTB acc | PTB bad |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 1 | quick | `mg_balanced_or_m_strong_detail_midbad__bw_badstrong__cw1p00_1p55_1p70` | 0 | 0.8232 | 0.7552 | 0.897/0.758/0.869 | 0.8317 | 0.9205 | 0.9932 |
| 2 | quick | `mg_qrs_confound_m_soft_midbad__bw_badstrong__cw1p00_1p55_1p70` | 0 | 0.7995 | 0.7266 | 0.795/0.805/0.783 | 0.7839 | 0.8783 | 0.9932 |
| 3 | seed | `mg_visible_unusable_m_soft_badguard__bw_badstrong__cw1p00_1p62_1p78` | 0 | 0.7470 | 0.7178 | 0.604/0.861/0.781 | 0.7540 | 0.9055 | 0.9850 |
| 4 | seed | `mg_balanced_or_m_soft_softbad__bw_badstrong__cw1p00_1p45_1p75` | 0 | 0.7792 | 0.7034 | 0.776/0.780/0.796 | 0.7791 | 0.8851 | 0.9986 |
| 5 | quick | `mg_visible_unusable_m_soft_badguard__bw_badstrong__cw1p00_1p62_1p78` | 0 | 0.7951 | 0.7446 | 0.767/0.820/0.774 | 0.7770 | 0.8774 | 0.9796 |
| 6 | seed | `mg_visible_unusable_m_soft_badguard__bw_badstrong__cw1p00_1p62_1p78` | 0 | 0.8057 | 0.7369 | 0.789/0.823/0.766 | 0.7757 | 0.8969 | 0.9973 |
| 7 | full | `mg_balanced_or_m_soft_softbad__bw_badstrong__cw1p00_1p45_1p75` | 0 | 0.7346 | 0.6856 | 0.651/0.795/0.822 | 0.7429 | 0.9114 | 0.9877 |
| 8 | quick | `mg_balanced_or_m_soft_softbad__bw_badstrong__cw1p00_1p45_1p75` | 0 | 0.7476 | 0.6819 | 0.686/0.798/0.752 | 0.7390 | 0.8906 | 0.9946 |
| 9 | quick | `mg_visible_unusable_m_strong_detail_midbad__bw_badstrong__cw1p00_1p55_1p70` | 0 | 0.7378 | 0.6714 | 0.732/0.742/0.745 | 0.7361 | 0.8924 | 0.9973 |
| 10 | quick | `mg_qrs_confound_m_strong_detail_softbad__bw_badstrong__cw1p00_1p45_1p75` | 0 | 0.8214 | 0.7519 | 0.896/0.767/0.745 | 0.7930 | 0.8978 | 0.9918 |
| 11 | full | `mg_visible_unusable_m_soft_badguard__bw_badstrong__cw1p00_1p62_1p78` | 0 | 0.7931 | 0.7238 | 0.859/0.731/0.873 | 0.8189 | 0.9042 | 0.9877 |
| 12 | quick | `mg_balanced_or_m_strong_detail_badguard__bw_badstrong__cw1p00_1p62_1p78` | 0 | 0.7795 | 0.7091 | 0.843/0.728/0.779 | 0.7773 | 0.8619 | 0.9905 |
| 13 | quick | `mg_qrs_confound_m_strong_detail_midbad__bw_badstrong__cw1p00_1p55_1p70` | 0 | 0.7553 | 0.6822 | 0.777/0.735/0.776 | 0.7449 | 0.8955 | 0.9891 |
| 14 | quick | `mg_visible_unusable_m_guard_badguard__bw_badstrong__cw1p00_1p62_1p78` | 0 | 0.7404 | 0.6689 | 0.791/0.693/0.800 | 0.7589 | 0.9024 | 0.9973 |
| 15 | seed | `mg_balanced_or_m_soft_softbad__bw_badstrong__cw1p00_1p45_1p75` | 0 | 0.6658 | 0.6497 | 0.584/0.714/0.876 | 0.7259 | 0.9124 | 0.9959 |
| 16 | full | `mg_qrs_confound_m_soft_midbad__bw_badstrong__cw1p00_1p55_1p70` | 0 | 0.8075 | 0.7562 | 0.897/0.739/0.754 | 0.7787 | 0.9092 | 0.9877 |
| 17 | seed | `mg_balanced_or_m_soft_softbad__bw_badstrong__cw1p00_1p45_1p75` | 0 | 0.7940 | 0.7256 | 0.899/0.707/0.808 | 0.8035 | 0.9096 | 0.9932 |
| 18 | quick | `mg_balanced_or_m_soft_badguard__bw_badstrong__cw1p00_1p62_1p78` | 0 | 0.7663 | 0.6816 | 0.847/0.695/0.815 | 0.7804 | 0.8733 | 0.9946 |
| 19 | full | `mg_visible_unusable_m_strong_detail_midbad__bw_badstrong__cw1p00_1p55_1p70` | 0 | 0.7639 | 0.6976 | 0.829/0.707/0.798 | 0.7667 | 0.9210 | 0.9891 |
| 20 | quick | `mg_balanced_or_m_soft_midbad__bw_badstrong__cw1p00_1p55_1p70` | 0 | 0.7826 | 0.7198 | 0.832/0.751/0.684 | 0.7509 | 0.8833 | 0.9905 |
| 21 | quick | `mg_visible_unusable_m_soft_midbad__bw_badstrong__cw1p00_1p55_1p70` | 0 | 0.7545 | 0.6837 | 0.738/0.777/0.662 | 0.7275 | 0.8669 | 0.9918 |
| 22 | quick | `mg_balanced_or_m_guard_badguard__bw_badstrong__cw1p00_1p62_1p78` | 0 | 0.7303 | 0.6592 | 0.780/0.676/0.876 | 0.7710 | 0.8674 | 0.9946 |
| 23 | quick | `mg_balanced_or_m_guard_softbad__bw_badstrong__cw1p00_1p45_1p75` | 0 | 0.8072 | 0.7415 | 0.784/0.839/0.672 | 0.7557 | 0.8837 | 0.9632 |
| 24 | quick | `mg_qrs_confound_m_soft_badguard__bw_badstrong__cw1p00_1p62_1p78` | 0 | 0.7202 | 0.6420 | 0.748/0.688/0.822 | 0.7386 | 0.8860 | 0.9973 |
| 25 | quick | `mg_visible_unusable_m_guard_midbad__bw_badstrong__cw1p00_1p55_1p70` | 0 | 0.7592 | 0.6903 | 0.854/0.677/0.803 | 0.7668 | 0.8783 | 0.9905 |
| 26 | seed | `mg_qrs_confound_m_soft_midbad__bw_badstrong__cw1p00_1p55_1p70` | 0 | 0.7565 | 0.6901 | 0.856/0.670/0.803 | 0.7722 | 0.9251 | 0.9986 |
| 27 | seed | `mg_qrs_confound_m_soft_midbad__bw_badstrong__cw1p00_1p55_1p70` | 0 | 0.7859 | 0.7145 | 0.900/0.690/0.808 | 0.7890 | 0.9105 | 0.9959 |
| 28 | quick | `mg_contact_lowamp_m_soft_softbad__bw_badstrong__cw1p00_1p45_1p75` | 0 | 0.7668 | 0.7034 | 0.871/0.682/0.757 | 0.7721 | 0.8724 | 0.9905 |
| 29 | quick | `mg_visible_unusable_m_soft_softbad__bw_badstrong__cw1p00_1p45_1p75` | 0 | 0.7480 | 0.6662 | 0.740/0.764/0.655 | 0.7158 | 0.8765 | 0.9946 |
| 30 | quick | `mg_contact_lowamp_m_soft_badguard__bw_badstrong__cw1p00_1p62_1p78` | 0 | 0.7005 | 0.6252 | 0.765/0.629/0.900 | 0.7511 | 0.8665 | 0.9823 |
| 31 | quick | `mg_contact_lowamp_m_strong_detail_badguard__bw_badstrong__cw1p00_1p62_1p78` | 0 | 0.7596 | 0.7066 | 0.894/0.655/0.701 | 0.7372 | 0.8883 | 0.9946 |
| 32 | full | `mg_balanced_or_m_strong_detail_midbad__bw_badstrong__cw1p00_1p55_1p70` | 0 | 0.7728 | 0.7022 | 0.956/0.615/0.852 | 0.8062 | 0.9296 | 0.9905 |
| 33 | quick | `mg_qrs_confound_m_strong_detail_badguard__bw_badstrong__cw1p00_1p62_1p78` | 0 | 0.7678 | 0.6925 | 0.786/0.780/0.482 | 0.6654 | 0.8992 | 0.9946 |
| 34 | quick | `mg_balanced_or_m_guard_midbad__bw_badstrong__cw1p00_1p55_1p70` | 0 | 0.7568 | 0.7028 | 0.720/0.787/0.757 | 0.7440 | 0.8892 | 0.9905 |
| 35 | quick | `mg_contact_lowamp_m_soft_midbad__bw_badstrong__cw1p00_1p55_1p70` | 0 | 0.6952 | 0.6140 | 0.794/0.594/0.912 | 0.7407 | 0.8651 | 0.9918 |
| 36 | quick | `mg_visible_unusable_m_strong_detail_badguard__bw_badstrong__cw1p00_1p62_1p78` | 0 | 0.7816 | 0.7217 | 0.842/0.737/0.730 | 0.7543 | 0.8969 | 0.9864 |
| 37 | full | `mg_qrs_confound_m_strong_detail_softbad__bw_badstrong__cw1p00_1p45_1p75` | 0 | 0.7414 | 0.7036 | 0.987/0.538/0.762 | 0.7505 | 0.9332 | 0.9932 |
| 38 | seed | `mg_visible_unusable_m_soft_badguard__bw_badstrong__cw1p00_1p62_1p78` | 0 | 0.6680 | 0.6277 | 0.498/0.786/0.912 | 0.7163 | 0.9033 | 0.9891 |
| 39 | seed | `mg_qrs_confound_m_soft_midbad__bw_badstrong__cw1p00_1p55_1p70` | 0 | 0.8067 | 0.7408 | 0.898/0.761/0.499 | 0.7112 | 0.9074 | 0.9837 |
