"""
Real-time insurance claim fraud scoring API.

Loads the trained artefacts ONCE at startup and reuses the project's own pipeline code:
    src/feature_engineering.py -> preprocess()     raw claim -> model-ready features
    src/train_model.py         -> score_claims()   anomaly score + IF flag + rule_score +
                                                   combined_risk_level (it calls
                                                   compute_rule_score() and assign_risk_level()
                                                   internally, so the API applies exactly the
                                                   logic that was evaluated in training)

Endpoints
    GET  /             basic API info
    GET  /health       liveness + whether the model is loaded
    POST /score_claim  score one claim
    POST /score_batch  score a list of claims
    GET  /docs         interactive Swagger UI (built into FastAPI)

Run from the project root:
    uvicorn api.main:app --reload

requirements.txt - add if not already listed:
    fastapi>=0.110
    uvicorn[standard]>=0.29
    pydantic>=2.0
    # already needed by src/: pandas, numpy, scikit-learn, joblib
"""

import json
import logging
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Literal, Optional

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# =============================================================================
# Paths (same PROJECT_ROOT pattern as the scripts in src/)
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent      # api/ -> project root
SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PKL = MODELS_DIR / "isolation_forest_model.pkl"
SCALER_PKL = MODELS_DIR / "scaler.pkl"
FEATURE_COLUMNS_JSON = MODELS_DIR / "feature_columns.json"

# Make src/ importable so the API reuses the pipeline instead of duplicating it
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from feature_engineering import CLAIM_TYPES, preprocess  # noqa: E402
from train_model import RULE_COLUMNS, score_claims  # noqa: E402

# =============================================================================
# Configuration
# =============================================================================
API_NAME = "Insurance Claim Fraud Scoring API"
API_VERSION = "1.0.0"
MAX_BATCH_SIZE = 1_000
RISK_LEVELS = ["High", "Medium", "Low"]

# Human-readable names for the binary rule flags used in the explanation string
RULE_DESCRIPTIONS = {
    "is_early_policy_claim": "early policy claim",
    "is_near_limit": "near coverage limit",
    "is_high_frequency_claimant": "high claim frequency (3+ in 6 months)",
    "is_delayed_reporting": "delayed reporting (15+ days after incident)",
    "any_recent_change": "recent address/bank details change",
    "is_round_number": "round claim amount",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("fraud_api")

# Example shown in Swagger UI (/docs), also used as a startup smoke test
EXAMPLE_CLAIM = {
    "claim_id": "CLM_DEMO_001",
    "policy_renewal_date": None,
    "claim_filed_date": "2026-08-14",
    "claim_type": "health",
    "claim_amount": 285000,
    "policy_coverage_limit": 300000,
    "claimant_avg_past_claim": None,
    "claimant_claim_count_6m": 1,
    "hospital_garage_code": "HSP0042",
    "claimant_address_changed_recently": False,
    "claimant_bank_details_changed_recently": True,
    "is_weekend_filed": False,
    "days_policy_to_claim": 22,
    "days_incident_to_claim": 18,
}


# =============================================================================
# Request / response schemas
# =============================================================================
class ClaimInput(BaseModel):
    """Raw fields for one claim - the same columns as REQUIRED_INPUT_COLUMNS in
    feature_engineering.py. Missing or wrongly typed fields are rejected with a 400."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="ignore",                                   # tolerate extra fields from callers
        json_schema_extra={"examples": [EXAMPLE_CLAIM]},
    )

    claim_id: str = Field(..., min_length=1, description="Unique claim identifier")
    policy_renewal_date: Optional[date] = Field(
        None, description="Most recent renewal on/before the incident; null if still in first term")
    claim_filed_date: date = Field(..., description="Date the claim was filed (YYYY-MM-DD)")
    claim_type: str = Field(..., description=f"One of: {', '.join(CLAIM_TYPES)}")
    claim_amount: float = Field(..., gt=0, description="Claimed amount (INR)")
    policy_coverage_limit: float = Field(..., gt=0, description="Policy sum insured (INR)")
    claimant_avg_past_claim: Optional[float] = Field(
        None, ge=0, description="Average of the claimant's previous claims; null for first-time claimants")
    claimant_claim_count_6m: int = Field(
        ..., ge=0, description="Claims by this claimant in the last 6 months, including this one")
    hospital_garage_code: Optional[str] = Field(
        None, description="Provider code; null/empty if unknown")
    claimant_address_changed_recently: bool
    claimant_bank_details_changed_recently: bool
    is_weekend_filed: bool
    days_policy_to_claim: int = Field(..., ge=0, description="claim_filed_date - policy_start_date")
    days_incident_to_claim: int = Field(..., ge=0, description="claim_filed_date - incident_date")

    @field_validator("claim_type")
    @classmethod
    def claim_type_must_be_trained_category(cls, value: str) -> str:
        normalised = value.lower()
        if normalised not in CLAIM_TYPES:
            raise ValueError(f"claim_type '{value}' is not supported; "
                             f"the model was trained on: {', '.join(CLAIM_TYPES)}")
        return normalised

    @field_validator("hospital_garage_code")
    @classmethod
    def blank_provider_is_unknown(cls, value: Optional[str]) -> Optional[str]:
        return value or None          # "" -> None -> preprocess() fills it with 'UNKNOWN'

    @model_validator(mode="after")
    def dates_must_be_consistent(self):
        if self.policy_renewal_date and self.policy_renewal_date > self.claim_filed_date:
            raise ValueError("policy_renewal_date cannot be after claim_filed_date")
        if self.days_incident_to_claim > self.days_policy_to_claim:
            raise ValueError("days_incident_to_claim cannot exceed days_policy_to_claim "
                             "(the incident would be before the policy started)")
        return self


class ClaimScore(BaseModel):
    claim_id: str
    anomaly_score: float = Field(..., description="Isolation Forest score; higher = more unusual")
    if_anomaly: bool = Field(..., description="True if above the contamination threshold")
    rule_score: int = Field(..., ge=0, le=len(RULE_COLUMNS), description="Number of rule flags triggered")
    combined_risk_level: Literal["High", "Medium", "Low"]
    triggered_rules: List[str]
    explanation: str


class BatchScoreResponse(BaseModel):
    count: int
    risk_level_counts: Dict[str, int]
    results: List[ClaimScore]


# =============================================================================
# Artefact loading (once, at startup)
# =============================================================================
@dataclass
class ModelArtifacts:
    model: object
    scaler: object
    feature_columns: List[str]


def load_artifacts() -> ModelArtifacts:
    """Load model, scaler and feature config, check they agree, then smoke-test the pipeline."""
    missing = [str(p) for p in (MODEL_PKL, SCALER_PKL, FEATURE_COLUMNS_JSON) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing model artefacts: {missing}")

    model = joblib.load(MODEL_PKL)
    scaler = joblib.load(SCALER_PKL)
    config = json.loads(FEATURE_COLUMNS_JSON.read_text())
    feature_columns = list(config["model_features"])

    # The model must have been trained on exactly these columns, in this order
    trained_on = list(getattr(model, "feature_names_in_", feature_columns))
    if trained_on != feature_columns:
        raise RuntimeError("feature_columns.json does not match the columns the model was trained on")

    # The scaler must expect the numeric columns listed in the config
    scaler_cols = list(getattr(scaler, "feature_names_in_", []))
    if scaler_cols and "numeric_features" in config and scaler_cols != list(config["numeric_features"]):
        raise RuntimeError("scaler.pkl does not match numeric_features in feature_columns.json")

    # The categories in the code must match the categories used when the model was built
    if "claim_types" in config and sorted(config["claim_types"]) != sorted(CLAIM_TYPES):
        raise RuntimeError(f"claim_types mismatch: config {config['claim_types']} vs code {CLAIM_TYPES}")

    # n_jobs=-1 was useful for training, but for one-row requests the parallel overhead
    # adds latency. Changing n_jobs does not change predictions.
    if hasattr(model, "n_jobs"):
        model.set_params(n_jobs=1)

    artifacts = ModelArtifacts(model=model, scaler=scaler, feature_columns=feature_columns)

    # Fail at startup (not on the first real request) if the pipeline is broken
    score_claim_inputs([ClaimInput(**EXAMPLE_CLAIM)], artifacts)
    return artifacts


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.artifacts = load_artifacts()
        logger.info("Model artefacts loaded from %s", MODELS_DIR)
    except Exception:
        # Keep the server up so /health can report the problem; scoring returns 503
        logger.exception("Failed to load model artefacts - scoring endpoints will return 503")
        app.state.artifacts = None
    yield


def get_artifacts(request: Request) -> ModelArtifacts:
    artifacts = getattr(request.app.state, "artifacts", None)
    if artifacts is None:
        raise HTTPException(status_code=503,
                            detail="Model artefacts are not loaded - check server logs and /health.")
    return artifacts


# =============================================================================
# Scoring logic (shared by /score_claim and /score_batch)
# =============================================================================
def claims_to_dataframe(claims: List[ClaimInput]) -> pd.DataFrame:
    """Validated Pydantic objects -> the raw DataFrame shape preprocess() expects."""
    raw = pd.DataFrame([claim.model_dump() for claim in claims])
    for col in ("policy_renewal_date", "claim_filed_date"):
        raw[col] = pd.to_datetime(raw[col])                       # None -> NaT
    raw["claimant_avg_past_claim"] = pd.to_numeric(raw["claimant_avg_past_claim"], errors="coerce")
    return raw


def build_features(raw: pd.DataFrame, artifacts: ModelArtifacts) -> pd.DataFrame:
    """Run the training-time preprocessing with the FITTED scaler (transform only, never refit)."""
    output = preprocess(raw, scaler=artifacts.scaler)
    # preprocess() returns (features, scaler, engineered); accept a bare DataFrame too
    features = output[0] if isinstance(output, tuple) else output
    missing = [c for c in artifacts.feature_columns if c not in features.columns]
    if missing:
        raise RuntimeError(f"preprocess() did not produce expected feature columns: {missing}")
    return features[artifacts.feature_columns]                    # exact training column order


def build_explanation(triggered: List[str], if_anomaly: bool) -> str:
    parts = [f"Flagged for: {', '.join(triggered)}" if triggered else "No rule flags triggered"]
    if if_anomaly:
        parts.append("Isolation Forest marked the claim as a statistical anomaly")
    return "; ".join(parts) + "."


def score_claim_inputs(claims: List[ClaimInput], artifacts: ModelArtifacts) -> List[ClaimScore]:
    """Score any number of claims in one vectorised pass."""
    raw = claims_to_dataframe(claims)
    X = build_features(raw, artifacts)

    # Same function used for evaluation in train_model.py: anomaly_score, if_anomaly,
    # rule_score (compute_rule_score) and combined_risk_level (assign_risk_level)
    scored = score_claims(artifacts.model, X, raw["claim_id"])

    rule_flags = X[RULE_COLUMNS].astype(int).to_numpy()
    results = []
    for i, row in enumerate(scored.itertuples(index=False)):
        triggered = [RULE_DESCRIPTIONS.get(col, col.replace("_", " "))
                     for col, is_on in zip(RULE_COLUMNS, rule_flags[i]) if is_on]
        results.append(ClaimScore(
            claim_id=str(row.claim_id),
            anomaly_score=round(float(row.anomaly_score), 4),
            if_anomaly=bool(row.if_anomaly),
            rule_score=int(row.rule_score),
            combined_risk_level=str(row.combined_risk_level),
            triggered_rules=triggered,
            explanation=build_explanation(triggered, bool(row.if_anomaly)),
        ))
    return results


def run_scoring(claims: List[ClaimInput], artifacts: ModelArtifacts) -> List[ClaimScore]:
    """Map pipeline errors to clean HTTP responses."""
    try:
        return score_claim_inputs(claims, artifacts)
    except ValueError as exc:          # e.g. preprocess() rejecting a category or missing column
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Scoring failed")
        raise HTTPException(status_code=500, detail="Internal error while scoring the claim(s).") from exc


# =============================================================================
# App & error handling
# =============================================================================
app = FastAPI(
    title=API_NAME,
    version=API_VERSION,
    description="Real-time fraud risk scoring: Isolation Forest anomaly detection + business rules.",
    lifespan=lifespan,
)

# Allow the local test frontend to call the API from the browser (development only).
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "null",                     # index.html opened directly as a file (file://)
        "http://127.0.0.1:5500",    # served locally, e.g. VS Code Live Server / python http.server
        "http://localhost:5500",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

ERROR_RESPONSES = {
    400: {"description": "Invalid input (missing field, wrong type, unsupported claim_type)"},
    503: {"description": "Model artefacts not loaded"},
}


def format_error_location(loc) -> str:
    """('body', 0, 'claim_type') -> '[0].claim_type'"""
    parts = []
    for item in loc:
        if item == "body":
            continue
        if isinstance(item, int):
            parts.append(f"[{item}]")
        else:
            parts.append(("." if parts else "") + str(item))
    return "".join(parts) or "body"


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """FastAPI returns 422 by default; return a clear 400 listing every problem instead."""
    errors = [{"field": format_error_location(err.get("loc", ())), "message": err.get("msg", "")}
              for err in exc.errors()]
    return JSONResponse(status_code=400, content={"detail": "Invalid claim input", "errors": errors})


# =============================================================================
# Endpoints
# =============================================================================
@app.get("/")
def root(request: Request):
    return {
        "name": API_NAME,
        "version": API_VERSION,
        "model_loaded": getattr(request.app.state, "artifacts", None) is not None,
        "docs": "/docs",
        "endpoints": {
            "GET /": "API info",
            "GET /health": "Health check",
            "POST /score_claim": "Score a single claim",
            "POST /score_batch": f"Score a list of claims (max {MAX_BATCH_SIZE})",
        },
    }


@app.get("/health")
def health(request: Request, response: Response):
    loaded = getattr(request.app.state, "artifacts", None) is not None
    if not loaded:
        response.status_code = 503        # lets load balancers / monitors detect the failure
    return {"status": "ok" if loaded else "unavailable", "model_loaded": loaded}


@app.post("/score_claim", response_model=ClaimScore, responses=ERROR_RESPONSES)
def score_claim(claim: ClaimInput, artifacts: ModelArtifacts = Depends(get_artifacts)):
    # Plain `def` (not async): FastAPI runs it in a thread pool, so CPU-bound
    # scikit-learn work does not block the event loop.
    return run_scoring([claim], artifacts)[0]


@app.post("/score_batch", response_model=BatchScoreResponse, responses=ERROR_RESPONSES)
def score_batch(claims: List[ClaimInput], artifacts: ModelArtifacts = Depends(get_artifacts)):
    if not claims:
        raise HTTPException(status_code=400, detail="Request body must contain at least one claim.")
    if len(claims) > MAX_BATCH_SIZE:
        raise HTTPException(status_code=400,
                            detail=f"Batch too large: {len(claims)} claims (max {MAX_BATCH_SIZE}).")

    results = run_scoring(claims, artifacts)
    counts = {level: 0 for level in RISK_LEVELS}
    for result in results:
        counts[result.combined_risk_level] += 1
    return BatchScoreResponse(count=len(results), risk_level_counts=counts, results=results)
