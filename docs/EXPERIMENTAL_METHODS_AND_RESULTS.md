# Methods and reported findings

## Experimental objective
The study evaluates an explainable, policy-constrained intrusion-detection decision pipeline for a proposed openEuler integration. A 12-feature public network-flow extract provides the independent data. The stages are input validation, leakage screening, binary classification, separate isotonic calibration, validation-only threshold selection, feature attribution diagnostics, risk-tier mitigation recommendations and data-noise stress checks.

## Model and study design
The deterministic split and model implementation are published in [the executable experiment](../code/UNSW_NB15_Extract_Experiment.mjs). Training uses 2,466 rows, calibration 616, threshold selection 411, and a held-out test of 619. The 160-stage boosted decision-stump implementation uses logistic loss with Newton-style leaf updates, L2 regularization of 6 and a step size of 0.085. It is a transparent custom boosted-stump model rather than the XGBoost library. A separate logistic-regression baseline is provided. Isotonic calibration is fitted on the calibration data alone; the frozen operating threshold is 0.75.

## Observed test outcomes
The logistic baseline yielded 88.05% accuracy, 72.59% F1 and 0.9275 AUROC. Boosted stumps yielded 93.54% accuracy, 86.11% F1 and 0.9666 AUROC. The final calibrated detector yielded 92.57% accuracy, 86.36% precision, 80.28% recall, 83.21% F1, 3.77% FPR and 0.9690 AUROC. Confusion counts at threshold 0.75 were 459 true negatives, 18 false positives, 28 false negatives and 114 true positives. Expected calibration error improved from 0.0480 to 0.0201; Brier score improved from 0.05927 to 0.05482.

## Explanation diagnostics
The custom boosted ensemble can be decomposed additively by each feature's contributions to individual stump margins. Training-stage split gains emphasize unique destination IPs, inter-arrival mean, packet length and unique destination ports. These are *model-dependence diagnostics*, not validated causal explanations. The reproducibility program writes a worked local additive decomposition to its output metrics.

## Offline response policies
No response, block-every-alert, score-only risk tiers and risk tiers with a feature-group evidence gate were compared using fixed intervention cost weights. The aggressive alert rule recommends actions for 80.28% of labelled attacks but also 3.77% of benign records. Score-only risk tiers recommend actions for 60.56% of attacks with 1.68% benign burden. The evidence gate reduces benign action recommendations to 0.21% but also reduces attack action coverage to **0.70%**, a substantial loss of utility. The assumed mean costs were 0.2294 (no response), 0.1073 (block-all), 0.1353 (risk-tier) and 0.2215 (evidence gate).

## Robustness and limitations
The program also tests additive standardized Gaussian feature noise at four predeclared noise levels, without altering thresholds. It does not run a separately measured real-world distribution-shift benchmark. All mitigation outcomes are outputs of an offline simulator. They do not establish safe block effectiveness, intrusion containment, service uptime, live false-positive control or measured openEuler throughput.

## Reproducibility
`node code/UNSW_NB15_Extract_Experiment.mjs` downloads and runs the public mapped extract. `python code/check_results.py` verifies counts and metrics; `python code/render_figures.py` regenerates nine plotted figures using the saved verified metrics. See [dataset provenance](DATASET_PROVENANCE.md) for lineage and leakage-screening details.
