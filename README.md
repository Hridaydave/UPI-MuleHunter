# MuleHunter-SAML v14

End-to-end research prototype for temporal transaction-graph money-mule detection.

## What v14 adds

- SAML-D research engine retained as the model layer.
- FastAPI inference/API layer.
- Web Control Tower dashboard.
- Transaction/account risk scoring interface.
- Ring detection: star, funnel, chain, cycle, dense neighborhood.
- Explainability endpoint with feature contributions.
- Metrics/results ingestion from the research experiments.
- SQLite prototype storage so the system runs without a database server.
- Docker Compose deployment.
- Clear separation between research experiments and serving/application code.

This project is **not a copy of the referenced GitHub MULE_HUNTER repository**. Its architecture is independently implemented around the SAML-D temporal transaction graph used in this project.

## Architecture

SAML-D
  -> temporal feature engine
  -> Markov / RF / GraphSAGE research models
  -> risk fusion
  -> graph ring intelligence
  -> explainability
  -> FastAPI
  -> Control Tower dashboard

## Quick start

### 1. Create environment

Windows PowerShell:

```powershell
python -m venv .venv
& ".\.venv\Scripts\Activate.ps1"
pip install -r requirements.txt
```

### 2. Start API

```powershell
uvicorn backend.app:app --reload
```

Open:

- http://127.0.0.1:8000/
- http://127.0.0.1:8000/docs

### 3. Start dashboard

The API serves the dashboard directly. Open:

http://127.0.0.1:8000/

## Load experiment results

Put exported result files under:

```text
results/
  metrics_comparison.csv
  profiling.json
  threshold_analysis.csv
  feature_importance.csv
```

Then:

```powershell
python scripts/import_results.py --results results
```

## Optional SAML-D experiment

The original research engine is under `models/`.

Run the research experiment separately from the serving layer:

```powershell
& ".\.venv\Scripts\python.exe" ".\models\MuleHunter_v13_RF_LEAKAGE_AUDIT.py" --data "archive.zip"
```

The exact SAML-D archive is intentionally not bundled because it is dataset material, not application source.

## Important research rule

Do not use the dashboard's demo scores as paper results. Paper results must come from the controlled SAML-D experiments with documented temporal splits and leakage audits.

## Next research upgrades

1. Temporal feature audit for all account aggregates.
2. GAT/attention graph encoder.
3. True online feature updates.
4. Model calibration.
5. Multi-seed confidence intervals.
6. Real-time stream ingestion.
