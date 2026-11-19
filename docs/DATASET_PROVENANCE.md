# Source-data lineage and validation protocol

The experiment takes its input from the publicly available, processed **UNSW-NB15-derived 12D mapped extract** in [PhantomNet](https://github.com/sriram21-09/PhantomNet). It does **not** import the original UNSW official 175,341/82,332 train/test files.

- Original primary dataset project: https://research.unsw.edu.au/projects/unsw-nb15-dataset
- Third-party mapped extract: https://github.com/sriram21-09/PhantomNet/blob/main/data/external_benchmarks/unsw_nb15_canonical_12d.csv
- Description of the third-party extraction: https://github.com/sriram21-09/PhantomNet/blob/main/docs/ML-8_DATASET_PROVENANCE.md
- GitHub source blob SHA (mapped CSV): `ddd6c902fffa493b70eb99c4dddb9b98f77439bb`

## Expected input schema
`packet_length, protocol_encoding, dst_port_class, src_port_ephemeral, event_rate_1m, burst_rate_10s, inter_arrival_mean, inter_arrival_std, packet_size_variance, payload_entropy, unique_dst_ips, unique_dst_ports, label`.

The source contains 5,000 rows and a 70:30 benign/attack sample balance (3,500:1,500). This is an upstream-curated mapped extract, **not a representative sample of production prevalence**. This repository does not claim the original 2015 collection has openEuler-specific telemetry or all nine attack labels.

## Data cleaning and partitions
Before partitioning, observation rows are grouped by their exact predictor tuple. Repeated predictor vectors are collapsed, and groups with conflicting labels are discarded. This removes 888 rows, leaving 4,112 unique consistently labelled observations. A fixed seeded stratified split then assigns 2,466 fitting, 616 isotonic calibration, 411 operating-threshold selection, and 619 final-test observations. The 619 test examples contain 477 benign and 142 intrusion records. This sample does not provide real source-system identifiers; therefore **the split is row-disjoint after feature deduplication, not site-disjoint or temporal-out-of-distribution validation**.

Training-only signed-log and standard scaling is applied to all partitions, with clipping to [-8,8]. A regularized logistic baseline and a 160-stump boosted logistic detector are fitted. The calibration map and 3%-target validation threshold are frozen before the independent test.

## Distribution and permitted use
To respect the upstream dataset's ownership and changing access conditions, source data are fetched by the script at runtime rather than rehosted here. Refer to the original dataset and third-party repository for their applicable academic-use conditions. The local dataset path argument supports offline reproduction. Avoid uploading sensitive live traffic to this public repository.

## Interpretation
Final metrics concern the 12D mapped extract, binary traffic labels and offline action *recommendations*. They are not full UNSW-NB15 benchmark scores, openEuler runtime benchmarks, controlled network packet mitigation, or validation of unseen attack families.
