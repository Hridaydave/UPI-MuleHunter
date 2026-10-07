# Dataset Documentation

## Dataset Strategy

UPI-MuleHunter uses multiple AML transaction datasets for development,
external validation, and temporal generalization.

| Dataset | Role |
|---|---|
| SAML-D | Primary development dataset |
| IBM Transactions for Anti Money Laundering | External validation |
| Time Series of Transactions - Money Laundering | Temporal/generalization validation |

## SAML-D

SAML-D is used as the primary development dataset for transaction-network
analysis and model experimentation.

The dataset contains transaction-level information including:

- timestamp
- sender account
- receiver account
- transaction amount
- payment currency
- received currency
- sender bank location
- receiver bank location
- payment type
- laundering label

## IBM AML

The IBM Transactions for Anti Money Laundering dataset is used as an
external validation source.

Its purpose is to evaluate whether patterns learned during development
generalize to a different transaction dataset.

## Time-Series AML

The time-series dataset is used to evaluate temporal generalization and
robustness under chronological evaluation.

## Dataset Processing

Datasets are processed independently through a schema-adaptation layer.

They are not blindly concatenated.

The general pipeline is:

Dataset
→ Schema Adapter
→ Cleaning
→ Feature Engineering
→ Model Input

## Data Privacy

Raw datasets are not included in this GitHub repository.

Users should obtain datasets from their original sources and follow their
respective licensing and usage conditions.
