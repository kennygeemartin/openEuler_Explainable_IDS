# Explainable AI-Driven Intrusion Detection and Adaptive Threat Mitigation for openEuler

This repository contains the **reproducible experiment** associated with the study *Development of an Explainable AI-Driven Intrusion Detection and Adaptive Threat Mitigation Framework for openEuler-Based Cloud and Enterprise Networks*.

## Dataset and experimental scope

The experiments use a **third-party UNSW-NB15-derived, mapped 12-feature CSV sample**, **not** the official full UNSW-NB15 training/testing splits or live openEuler telemetry. The original 5,000 records were reduced to **4,112 unique, consistently labelled observations** before splitting. All reported response-policy outcomes are **offline recommendations**, not executed firewall actions.

- [Public mapped dataset](https://github.com/sriram21-09/PhantomNet/blob/main/data/external_benchmarks/unsw_nb15_canonical_12d.csv)
- [Source provenance](https://github.com/sriram21-09/PhantomNet/blob/main/docs/ML-8_DATASET_PROVENANCE.md)
- [Original UNSW-NB15 project](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
- Source GitHub blob SHA: `ddd6c902fffa493b70eb99c4dddb9b98f77439bb`

## Run the experiment

Requires **Node.js 18 or newer**. No npm packages needed. From the repository root:

```bash
node code/UNSW_NB15_Extract_Experiment.mjs
```

The program downloads the public mapped sample. The complete automated reproduction (including the nine PNG figures and held-out prediction CSV) succeeded in [GitHub Actions](https://github.com/kennygeemartin/openEuler_Explainable_IDS/actions/workflows/reproduce.yml). Alternatively provide an existing CSV:

```bash
node code/UNSW_NB15_Extract_Experiment.mjs data/unsw_nb15_canonical_12d.csv
```

Outputs written to the current working directory:

- `UNSW_Benchmark_Metrics.json` — exhaustive evaluation metrics, calibration bins, feature ranking and feature-noise sensitivity
- `UNSW_Heldout_Predictions.csv` — row-indexed independent test predictions and alerts

The randomization seed is `20261119`. Splits are stratified after exact-feature deduplication and ambiguous-label removal, and the scaler is fitted **only on training rows**.

## Experiment configuration

| Partition | All records | Benign | Intrusion |
|---|---:|---:|---:|
| Fit | 2,466 | 1,902 | 564 |
| Calibration | 616 | 475 | 141 |
| Threshold validation | 411 | 317 | 94 |
| Independent test | 619 | 477 | 142 |
| **Total** | **4,112** | **3,171** | **941** |

Two detectors: ridge-regularized logistic baseline (220 iterations), and a 160-stage Newton-updated **boosted decision-stump** ensemble (learning rate 0.085, L2 = 6, 18 threshold candidates/feature). The boosted model is *not a package-level XGBoost implementation*, nor a multiclass classifier. Isotonic calibration uses only the calibration partition. The operating threshold `0.75` is selected exclusively from benign validation scores for a nominal 3% FPR target (actual validation FPR 3.15%).

## Independent test results

| Model/policy | Accuracy | Precision | Recall | F1 | FPR | AUROC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic baseline, 0.50 | 88.05% | 76.56% | 69.01% | 72.59% | 6.29% | 0.9275 |
| Boosted stumps, 0.50 | 93.54% | 84.93% | 87.32% | 86.11% | 4.61% | 0.9666 |
| Calibrated, frozen 0.75 | 92.57% | 86.36% | 80.28% | 83.21% | 3.77% | 0.9690 |

Frozen calibrated confusion matrix: **TN 459, FP 18, FN 28, TP 114**. Expected calibration error fell from **0.0480** to **0.0201**. AUC can change slightly after isotonic calibration because it creates score ties.

## Offline mitigation decision study

| Recommendation policy | Attack intervention coverage | Benign intervention rate |
|---|---:|---:|
| Block every alert | 80.28% | 3.77% |
| Risk tiers | 60.56% | 1.68% |
| Risk tiers with evidence gate | 0.70% | 0.21% |

The evidence gate reduces benign recommendations substantially, but in this particular experiment it also suppresses nearly all attack interventions. This trade-off is reported rather than concealed or tuned using the independent test labels.

## Repository contents

- `code/UNSW_NB15_Extract_Experiment.mjs` — full reproducible Node.js workflow
- `results/UNSW_Benchmark_Verified_Results.json` — recorded summary metrics
- `docs/EXPERIMENTAL_METHODS_AND_RESULTS.md` — research protocol and interpretation
- `docs/DATASET_PROVENANCE.md` — sample lineage, feature list, deduplication and evaluation restrictions
- `figures/` — nine committed PNG charts, reproducible using `python code/render_figures.py`
- `code/check_results.py` — reproducibility assertions comparing the current run against the verified results
- `code/render_figures.py` — regeneration of all nine figure PNG files
- `results/UNSW_Benchmark_Metrics.json` — complete numerical run output
- `results/UNSW_Heldout_Predictions.csv` — held-out labels, probabilities and alerts
- `.github/workflows/reproduce.yml` — automated full run, verification, figure generation and artifact publication
- `requirements/` — Python plotting dependencies; Node model execution is dependency-free

The formatted Word manuscript remains available as the separate research document supplied in the ChatGPT session. This repository contains the complete executable experiment, recorded metrics, held-out predictions, visualizations, provenance and an automated reproducibility workflow. The source data are fetched from their original publisher. Publication claims should follow the dataset and deployment scope documented here.

## Ethical/operational note

This is an offline, labelled network-flow analysis. It does not perform packet interception, attack traffic generation, live blocking, or modifications to openEuler hosts. Deployment adapters are proposed interfaces rather than measured production integrations.

## Citation

When using these results, cite the associated research manuscript and the public mapped-dataset repository with its primary UNSW-NB15 provenance.
