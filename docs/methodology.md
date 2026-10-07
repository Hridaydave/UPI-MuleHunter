# Research Methodology

## Overview

UPI-MuleHunter combines behavioral, temporal and graph-based signals for
mule-account risk assessment.

## Behavioral Layer

The behavioral layer extracts transaction and account-level characteristics,
including transaction activity, amount patterns, counterparty behaviour and
cross-bank/cross-currency activity.

## Temporal Layer

A Markov-based model represents transitions between observed transaction
behaviour states.

The resulting temporal signal is used as one component of the overall
risk assessment.

## Graph Layer

The transaction ecosystem is represented as a directed graph.

- Account = Node
- Transaction = Edge
- Transaction attributes = Edge features

GraphSAGE is used to learn account representations from graph structure.

## Risk Fusion

Outputs from the behavioral, temporal and graph components can be combined
to produce a unified risk assessment.

## Evaluation

Evaluation should use:

- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Confusion Matrix

For imbalanced fraud/AML datasets, PR-AUC and recall are particularly
important.

## Leakage Control

Before reporting research results, the following must be verified:

1. Features are available at prediction time.
2. Future information does not enter training features.
3. Aggregation windows are correctly defined.
4. Labels are not encoded directly or indirectly in features.
5. Chronological evaluation is used where appropriate.
6. External validation is performed where possible.

## Research Principle

A very high metric is not automatically evidence of a strong model.

Performance must be checked for leakage, temporal validity and
cross-dataset generalization.
