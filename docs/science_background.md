# Scientific Background

## Why ECG quality is a modelling problem

An ECG analysis system can return a confident result even when motion
artefact, poor electrode contact, baseline drift, saturation, or signal loss
has removed the morphology needed for interpretation. Signal-quality
assessment therefore acts as an input-validity check before heart-rate,
rhythm, or morphology analysis.

This project follows the classical idea that no single quality mechanism is
sufficient. Instead, multiple imperfect indicators are combined into a quality
decision. The reproduced study is Clifford et al. [1](references.md#1-clifford-et-al-2012).

## Seven SQI families

For a 12-lead record, each SQI is computed per lead, producing an
84-dimensional representation.

| SQI | Mechanism represented | Typical failure signal |
|---|---|---|
| `bSQI` | Agreement between independent QRS detectors | Missed or inconsistent beats |
| `iSQI` | Inter-lead agreement | Lead-specific corruption |
| `kSQI` | Kurtosis | Abnormal waveform-shape distribution |
| `sSQI` | Skewness | Asymmetric shape distortion |
| `pSQI` | QRS-band spectral concentration | Spectral contamination |
| `fSQI` | Flat-line occupancy | Signal loss or saturation |
| `basSQI` | Low-frequency power | Baseline wander |

The fusion view can be written as

$$
z = \phi_{\mathrm{SQI}}(x), \qquad
P(Y=\mathrm{poor}\mid x) = F(z_1,\ldots,z_K).
$$

The individual features remain interpretable, while the classifier learns how
their failure modes overlap.

## Model families

**RBF-SVM.** A nonlinear margin classifier used for individual SQIs and their
predefined combinations. It is the closest classical comparison with the
paper.

**LM-MLP.** A small multilayer perceptron trained with a
Levenberg-Marquardt-style procedure. It tests whether flexible feature fusion
changes the classical conclusion.

**ResNet and Conformer.** Waveform models preserve local temporal evidence.
The matched ResNet is an architecture control: on BUT QDB it remained
comparable with the Conformer, preventing a universal architecture-specific
claim.

## Datasets and evaluation

| Dataset | Role | Labels used |
|---|---|---|
| PhysioNet/CinC 2011 Set-A | Public 12-lead reproduction and extension | Acceptable / unacceptable |
| MIT-BIH Noise Stress Test Database | Electrode-motion and muscle-artefact controls | Noise recordings |
| PTB-XL | Clean carrier signals for train-only proposal construction | Public waveform data |
| BUT QDB | Native single-lead graded evaluation | Good / Medium / Bad |

All augmentation is restricted to training. Validation and test partitions
remain native. Accuracy is reported with class recall, balanced accuracy,
ROC-AUC, PR-AUC, and macro-F1 as appropriate; these complementary metrics
separate ranking quality from a selected operating point.

## Validity principle

Balancing the class prior, $P(Y)$, does not guarantee recovery of the poor
signal distribution, $P(X\mid Y=\mathrm{poor})$. The project therefore audits
whether generated and native poor ECGs can be distinguished and whether models
trained on one source transfer to the other.
