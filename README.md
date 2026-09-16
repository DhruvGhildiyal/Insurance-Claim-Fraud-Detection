# Insurance Claim Fraud Detection: Isolation Forest + Business Rules

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-IsolationForest-F7931E?logo=scikitlearn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-real--time%20scoring-009688?logo=fastapi&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-data%20pipeline-150458?logo=pandas&logoColor=white)

> An end-to-end fraud scoring system: synthetic data → leakage-safe features → anomaly detection + business rules → a real-time API with a browser test UI.
>
> **Headline result:** claims in the **High** risk bucket are fraud **52.9%** of the time — a **10.6× lift** over the 5% base rate — while flagging only **3.4%** of claims for review.

---

## Overview

This project flags potentially fraudulent insurance claims by combining an **unsupervised Isolation Forest** with **transparent business rules**, then serves the result through a **FastAPI** endpoint that scores a claim in real time.

It's a strong ML use case because it has all the hard parts of real-world modeling:

- **Rare-event detection**: only ~5% of claims are fraudulent, so accuracy is a useless metric.
- **Imbalanced data**: the model is judged on PR-AUC and precision/recall, not a single headline score.
- **Few labels in practice**: an anomaly detector trains without labels, which matches how fraud data actually arrives (late, incomplete, and disputed).
- **Explainability**: investigators get the specific red flags behind every score, not just a number.

**The business framing:** Special Investigation Units can only review a small share of claims, so the real problem is **prioritisation** — putting the most suspicious claims in front of investigators first without drowning them in false alarms.

## What's in This Repo

| Folder / File | Contents | Purpose |
|---|---|---|
| `data/raw/` | `insurance_claims.csv`, `provider_master.csv` | Synthetic 10,000-claim dataset (5% fraud) plus a 200-provider lookup table. This is the input to the pipeline. |
| `data/processed/` | *(created when you run the pipeline)* | Model-ready features and a separate labels file. Not committed — regenerate with `feature_engineering.py`. |
| `models/` | `isolation_forest_model.pkl`, `scaler.pkl`, `feature_columns.json` | **Pre-trained artifacts — already trained and ready to use.** The API loads these at startup. |
| `reports/figures/` | 6 PNG charts | Evaluation visualizations (ROC, PR, confusion matrix, risk buckets, score distribution, feature correlation). Already generated. |
| `source_code/` | `generate_data.py`, `feature_engineering.py`, `train_model.py`, `visualize_results.py` | The full pipeline: data generation → feature engineering → training/evaluation → charts. |
| `api/` | `main.py` | FastAPI backend serving the trained model: `/score_claim`, `/score_batch`, `/health`. |
| `frontend/` | `index.html` | Single-file browser UI to test the API through a form — no JSON or code required. |
| `requirements.txt` | Python dependencies | Everything needed for the pipeline and the API. |
| `LICENSE` | License terms | Usage terms for this repo. |

## What You Get by Running the Code

**The model is already trained.** `models/` contains the fitted Isolation Forest, the fitted scaler and the feature column config, so:

> ✅ **To use the API, you do NOT need to run any training scripts.** Install the dependencies and start the server — skip straight to [Quickstart Option A](#option-a-just-test-the-api-fastest).

You only need the pipeline scripts if you want to **reproduce the project from scratch** or change something (different data, different features, different hyperparameters).

### Reproducing the full pipeline

Run these four scripts **in this exact order** — each one depends on the output of the previous:

| # | Script | Reads | Writes |
|---|---|---|---|
| 1 | `source_code/generate_data.py` | — | `data/raw/insurance_claims.csv`, `data/raw/provider_master.csv` |
| 2 | `source_code/feature_engineering.py` | `data/raw/` | `data/processed/` (features + labels), `models/scaler.pkl`, `models/feature_columns.json` |
| 3 | `source_code/train_model.py` | `data/processed/` | `models/isolation_forest_model.pkl` + the metrics printed to your terminal |
| 4 | `source_code/visualize_results.py` | `data/processed/`, `models/` | The 6 PNGs in `reports/figures/` |

> ⚠️ **This overwrites the files already in the repo** — everything in `data/`, `models/` and `reports/figures/`. Every script uses a fixed random seed (42), so you should get the same dataset, the same model and the same metrics shown below, as long as your library versions match `requirements.txt`.

## Results

All metrics are on a **held-out, stratified 20% test set**: 2,000 claims, 100 of them fraud (5.00%). The model was trained on the other 8,000 claims **without labels**.

### Summary metrics

| Metric | Value | Notes |
|---|---|---|
| ROC-AUC (Isolation Forest) | **0.862** | Threshold-free ranking quality (random = 0.500) |
| PR-AUC (Isolation Forest) | **0.361** | 7.2× the random baseline of 0.050 |
| Precision @ contamination 0.05 | **0.427** | Of 96 flagged claims, 41 were fraud |
| Recall @ contamination 0.05 | **0.410** | 41 of 100 fraud cases caught |
| F1 @ contamination 0.05 | **0.418** | — |
| ROC-AUC (rule score baseline) | **0.918** | The 6 business rules on their own |
| **High risk bucket lift** | **10.6×** | 52.9% of High claims are fraud vs the 5% base rate |

### Risk buckets

| Risk level | Claims | Fraud | Fraud rate | Share of all fraud |
|---|---|---|---|---|
| 🔴 **High** (model **and** rules) | 68 (3.4%) | 36 | **52.9%** | 36.0% |
| 🟠 **Medium** (one signal only) | 101 (5.1%) | 34 | 33.7% | 34.0% |
| ⚪ **Low** (neither) | 1,831 (91.6%) | 30 | 1.6% | 30.0% |

**Reviewing just 3.4% of claims catches 36% of all fraud.** Widening to High + Medium (8.5% of claims) catches **70%**.

### ROC Curve
![ROC curve](reports/figures/roc_curve.png)
*Both methods rank fraud well above random: the rule score reaches ROC-AUC 0.918 and the Isolation Forest 0.862. The marker shows the model's operating point at the 5% contamination threshold.*

### Precision-Recall Curve
![Precision-recall curve](reports/figures/precision_recall_curve.png)
*PR-AUC of 0.361 is 7.2× the 5% random baseline. The diamond marks the High risk bucket at 52.9% precision — the highest precision of any strategy tested.*

### Confusion Matrix
![Confusion matrix](reports/figures/confusion_matrix.png)
*At the 5% threshold the model catches 41 of 100 fraud cases, sends 55 genuine claims (2.9% of all genuine claims) to review, and misses 59 fraud cases.*

### Risk Bucket Analysis
![Risk bucket analysis](reports/figures/risk_bucket_analysis.png)
*Only 68 of 2,000 claims land in High, but 52.9% of them are fraud. Low holds 1,831 claims at a 1.6% fraud rate, so most of the book can be safely deprioritised.*

### Anomaly Score Distribution
![Anomaly score distribution](reports/figures/anomaly_score_distribution.png)
*Fraud claims skew toward higher anomaly scores, but the distributions overlap substantially. That overlap is why the model alone reaches only 42.7% precision, and why pairing it with rules helps.*

### Feature Correlation with Fraud
![Feature correlation](reports/figures/feature_correlation.png)
*The strongest signals are the near-coverage-limit flag (+0.316; 40.2% of flagged claims are fraud) and round claim amounts (+0.303; 29.8%). Calendar features sit near zero. Analysis only — never used for training.*

## Key Engineering Decisions

> A model is only as trustworthy as the pipeline around it. These four decisions are the ones worth discussing in an interview.

- **🔒 Data leakage prevention**
  - `is_fraud` is **never passed to `model.fit()`**. Features and labels live in **separate files**, joined on `claim_id` only at evaluation time.
  - `provider_fraud_rate` is derived from labels, so it is computed with **out-of-fold encoding** (each claim's value comes from the other folds) and kept **out of the model inputs**. The naive in-sample version correlates **+0.323** with fraud versus **+0.275** out-of-fold — that gap is exactly the leakage the method removes.
  - Claimant history features (`claimant_avg_past_claim`, `claimant_claim_count_6m`) use **strictly earlier claims only**, so nothing looks ahead in time.

- **🎯 Why `contamination=0.05`**
  It matches the known fraud rate, so the number of flagged claims is about the number of real frauds, which makes precision and recall directly comparable. The threshold is learned on the training split (5.00% flagged on train, 4.80% on test). In production this is a **business lever**, not a learned parameter: set it by investigator capacity, trading recall against review workload.

- **📊 Why PR-AUC matters alongside ROC-AUC**
  The model scores a respectable **ROC-AUC of 0.862**, yet only **42.7%** of what it flags is fraud. With 5% positives, ROC-AUC flatters the model because the huge genuine class dominates the false-positive rate. **PR-AUC (0.361)** measures the rare class directly, against a random baseline of 0.05 rather than 0.5.

- **🤝 Why hybrid rules + ML, not ML alone**
  - On this data, **rules alone rank fraud better** than the Isolation Forest (ROC-AUC **0.918** vs **0.862**). That's expected and worth saying out loud: the rules encode the same patterns that were injected into the synthetic data, while the forest has to discover "unusual" with no labels at all.
  - The two signals are **complementary**. Requiring both for "High" pushes precision to **52.9%**, above rules alone (46.1%) and the model alone (42.7%).
  - Accepting either signal ("High" + "Medium") lifts recall to **70%** — a tiered review queue instead of one yes/no flag.
  - Rules give investigators a **readable reason** for every flag; the label-free model is the part that can surface suspicious claims nobody wrote a rule for.

## Quickstart

**Prerequisites:** Python 3.10+ and Git. Check with `python --version` (on some systems, `python3 --version`).

### Setup (both options)

**1. Clone the repo and enter it**

```bash
git clone https://github.com/<your-username>/insurance-fraud-detection.git
```

```bash
cd insurance-fraud-detection
```

**2. Create a virtual environment**

Windows (PowerShell):

```powershell
python -m venv .venv
```

```powershell
.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
python3 -m venv .venv
```

```bash
source .venv/bin/activate
```

Your prompt should now start with `(.venv)`.

**3. Install the dependencies**

```bash
pip install -r requirements.txt
```

---

### Option A: Just Test the API (fastest)

**No training required — the model in `models/` is already trained.**

**1. Start the API server** (run this from the project root, not from inside `api/`):

```bash
uvicorn api.main:app --reload
```

You should see `Uvicorn running on http://127.0.0.1:8000`. Leave this terminal open.

**2. Check that the model loaded.** Open http://127.0.0.1:8000/health in a browser. You want:

```json
{"status": "ok", "model_loaded": true}
```

**3. Score a claim.** Either:

- **Browser UI** — open `frontend/index.html` (double-click it, or drag it into a browser window). Click **Load Sample: Suspicious Claim**, then **Score This Claim**.
- **Swagger docs** — go to http://127.0.0.1:8000/docs, expand `POST /score_claim`, click **Try it out**, then **Execute**. The example claim is pre-filled.

> **If the browser UI shows "API unreachable" while the server is running**, it's CORS: the page and the API are on different origins. Add this to `api/main.py` just after `app = FastAPI(...)`:
>
> ```python
> from fastapi.middleware.cors import CORSMiddleware
>
> app.add_middleware(
>     CORSMiddleware,
>     allow_origins=["null", "http://127.0.0.1:5500", "http://localhost:5500"],
>     allow_methods=["GET", "POST"],
>     allow_headers=["Content-Type"],
> )
> ```
>
> Then restart the server. (`"null"` covers pages opened directly as files — development only.) The Swagger UI at `/docs` works either way, since it's served by the API itself.

---

### Option B: Reproduce the Full Pipeline

Runs everything from scratch and **overwrites** `data/`, `models/` and `reports/figures/`. Run all four commands from the **project root**, in this order:

**1. Generate the synthetic dataset** (10,000 claims, ~5% fraud):

```bash
python src/generate_data.py
```

**2. Build the features** (writes `data/processed/`, `models/scaler.pkl`, `models/feature_columns.json`):

```bash
python src/feature_engineering.py
```

**3. Train and evaluate the model** (writes `models/isolation_forest_model.pkl` and prints the metrics table):

```bash
python src/train_model.py
```

**4. Regenerate the charts** (writes the 6 PNGs in `reports/figures/`):

```bash
python src/visualize_results.py
```

Then follow Option A to serve the freshly trained model.

**Troubleshooting**

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'src'` or file-not-found errors | Run the commands from the **project root**, not from inside `src/`. |
| `ERROR: [Errno 10048] address already in use` | Port 8000 is taken. Use `uvicorn api.main:app --reload --port 8001` and update the URL in `frontend/index.html`. |
| `/health` returns `"model_loaded": false` | The files in `models/` are missing or unreadable. Check the uvicorn terminal output, or rebuild them with Option B steps 2-3. |
| `pip install` fails on a package build | Upgrade pip first: `python -m pip install --upgrade pip`. |

## API Usage Example

**Endpoints:** `POST /score_claim` · `POST /score_batch` · `GET /health` · `GET /` · `GET /docs`

### curl

```bash
curl -X POST http://127.0.0.1:8000/score_claim -H "Content-Type: application/json" -d '{"claim_id":"CLM_TEST_SUS_001","claim_type":"health","claim_amount":290000,"policy_coverage_limit":300000,"claimant_avg_past_claim":null,"claimant_claim_count_6m":1,"hospital_garage_code":"HSP0042","policy_renewal_date":null,"claim_filed_date":"2026-09-10","days_policy_to_claim":21,"days_incident_to_claim":17,"claimant_address_changed_recently":false,"claimant_bank_details_changed_recently":true,"is_weekend_filed":false}'
```

*(On Windows PowerShell, use `curl.exe` instead of `curl`, or use the Python example below.)*

### Python

```python
import requests

claim = {
    "claim_id": "CLM_TEST_SUS_001",
    "claim_type": "health",                  # auto | health | property
    "claim_amount": 290000,
    "policy_coverage_limit": 300000,         # claim is at 96.7% of the limit
    "claimant_avg_past_claim": None,         # None = first-time claimant
    "claimant_claim_count_6m": 1,
    "hospital_garage_code": "HSP0042",
    "policy_renewal_date": None,             # None = never renewed
    "claim_filed_date": "2026-09-10",
    "days_policy_to_claim": 21,              # filed 3 weeks after buying the policy
    "days_incident_to_claim": 17,            # reported 17 days after the incident
    "claimant_address_changed_recently": False,
    "claimant_bank_details_changed_recently": True,
    "is_weekend_filed": False,
}

response = requests.post("http://127.0.0.1:8000/score_claim", json=claim, timeout=10)
response.raise_for_status()
print(response.json())
```

### Expected response

*(Illustrative — the exact `anomaly_score`, and therefore the risk level, depend on the trained model.)*

```json
{
  "claim_id": "CLM_TEST_SUS_001",
  "anomaly_score": 0.5812,
  "if_anomaly": true,
  "rule_score": 5,
  "combined_risk_level": "High",
  "triggered_rules": [
    "early policy claim",
    "near coverage limit",
    "delayed reporting (15+ days after incident)",
    "recent address/bank details change",
    "round claim amount"
  ],
  "explanation": "Flagged for: early policy claim, near coverage limit, delayed reporting (15+ days after incident), recent address/bank details change, round claim amount; Isolation Forest marked the claim as a statistical anomaly."
}
```

**Error handling:** invalid input (an unsupported `claim_type`, a missing field, a wrong type) returns **400** with a list of the specific problem fields. If the model artifacts failed to load, scoring returns **503**.

## Tech Stack

| Area | Tools |
|---|---|
| Language | Python 3.10+ |
| Data | pandas, numpy, Faker (synthetic data generation) |
| Modeling | scikit-learn — `IsolationForest`, `StandardScaler`, `StratifiedKFold` |
| Visualization | matplotlib, seaborn |
| API | FastAPI, uvicorn, Pydantic |
| Frontend | Plain HTML / CSS / JavaScript (no framework, no build step) |
| Persistence | joblib (model + scaler artifacts) |

## Future Improvements

- **Supervised model comparison** — rules already out-rank the unsupervised model (0.918 vs 0.862 ROC-AUC), so benchmarking XGBoost/LightGBM with out-of-fold provider encoding is the natural next step.
- **Feature selection** — drop near-zero-signal inputs (`claim_month`, `claim_day_of_week`, `has_been_renewed`) that add noise to the Isolation Forest.
- **Catch the weak cases** — 30% of fraud still lands in the Low bucket; weighted rule scores or a supervised layer could recover single-flag fraud.
- **SHAP explainability** — per-claim feature attributions alongside the rule flags.
- **Database integration** — compute claimant-history features server-side from a claims database instead of trusting the API caller.
- **Stricter evaluation** — fit the scaler inside a scikit-learn `Pipeline` on the train split only, and add a time-based split.
- **API security** — API-key or OAuth2 authentication, rate limiting, request logging.
- **Deployment & MLOps** — Docker image, cloud deployment, CI tests, model versioning and drift monitoring.
