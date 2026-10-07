from pathlib import Path
import csv
import json
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .store import init_db, save_result, get_result
from .risk_fusion import fuse, risk_band
from .explain import explain
from .ring_detector import detect_rings

app = FastAPI(
    title="MuleHunter-SAML",
    version="14.0",
    description="Temporal transaction-graph mule detection research prototype",
)

init_db()

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"


class ScoreRequest(BaseModel):
    markov: float = Field(0.0, ge=0, le=1)
    rf: float = Field(0.0, ge=0, le=1)
    graph: float = Field(0.0, ge=0, le=1)
    anomaly: float = Field(0.0, ge=0, le=1)
    ring: float = Field(0.0, ge=0, le=1)
    features: dict = {}


class RingRequest(BaseModel):
    edges: list[dict]


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "service": "MuleHunter-SAML", "version": "14.0"}


@app.post("/api/score")
def score(req: ScoreRequest):
    fused = fuse(
        req.markov,
        req.rf,
        req.graph,
        req.anomaly,
        req.ring,
    )
    return {
        "markov": req.markov,
        "rf": req.rf,
        "graph": req.graph,
        "anomaly": req.anomaly,
        "ring": req.ring,
        "risk_score": fused,
        "risk_band": risk_band(fused),
        "reasons": explain(
            req.features,
            req.markov,
            req.rf,
            req.graph,
            req.ring,
        ),
    }


@app.post("/api/rings")
def rings(req: RingRequest):
    return {
        "count": len(detect_rings(req.edges)),
        "findings": detect_rings(req.edges),
    }


@app.post("/api/results/import")
def import_results(path: str = Query(...)):
    p = Path(path)
    if not p.exists():
        raise HTTPException(404, f"Path not found: {path}")

    imported = []

    for f in p.iterdir():
        if f.suffix.lower() == ".json":
            try:
                payload = json.loads(f.read_text(encoding="utf-8"))
                save_result(f.stem, payload)
                imported.append(f.name)
            except Exception:
                pass

        elif f.suffix.lower() == ".csv":
            try:
                rows = list(csv.DictReader(f.open("r", encoding="utf-8")))
                save_result(f.stem, rows)
                imported.append(f.name)
            except Exception:
                pass

    return {"imported": imported, "count": len(imported)}


@app.get("/api/results/{name}")
def result(name: str):
    payload = get_result(name)
    if payload is None:
        raise HTTPException(404, "Result not found")
    return payload
