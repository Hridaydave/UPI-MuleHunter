# Evaluation Protocol

## Purpose

This document defines the evaluation protocol for UPI-MuleHunter.

The objective is to determine whether behavioral, temporal and graph-based
signals provide useful mule/AML risk information while controlling for
feature leakage and temporal contamination.

---

## Evaluation Metrics

The following metrics are used where applicable:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Confusion Matrix

For highly imbalanced AML/fraud datasets, PR-AUC, precision, recall and
F1-score should receive particular attention.

---

## Dataset Evaluation Strategy

The project uses a multi-dataset evaluation strategy.

```text
                  SAML-D
                    |
                    v
             Development /
             Model Research
                    |
                    v
        +-----------+-----------+
        |                       |
        v                       v
   IBM AML                 Time-Series AML
 External Validation       Temporal Validation
