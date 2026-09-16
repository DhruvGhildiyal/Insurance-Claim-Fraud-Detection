# Insurance Claim Fraud Detection: Isolation Forest + Business Rules

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-IsolationForest-F7931E?logo=scikitlearn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-real--time%20scoring-009688?logo=fastapi&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-data%20pipeline-150458?logo=pandas&logoColor=white)

> An end-to-end fraud scoring system: synthetic data → leakage-safe features → anomaly detection + rules → a real-time API with a test UI.
>
> **Headline result:** claims in the **High** risk bucket are fraud **52.9%** of the time, a **10.6× lift** over the 5% base rate, while flagging only **3.4%** of claims for review.

---

## Overview

This project flags potentially fraudulent insurance claims by combining an **unsupervised Isolation Forest** with **transparent business rules**, then exposes the result through a **FastAPI** scoring service. Fraud detection is a strong ML use case: fraud is a **rare event (~5% of claims)**, labels are scarce and delayed in practice, and investigators need **explainable** decisions, not just a score.

## Problem Statement

Insurance fraud costs insurers billions every year, and those costs are passed on to honest policyholders as higher premiums. Special Investigation Units (SIUs) can only review a small fraction of claims, so the real business problem is **prioritisation**: put the most suspicious claims in front of investigators first, without flooding them with false alarms.

This project frames fraud detection the way it works in practice:

- **Imbalanced data**: accuracy is meaningless when 95% of claims are genuine.
- **Few reliable labels**: an anomaly detector that doesn't need labels to train is a realistic starting point.
- **Explainability matters**: every score comes with the specific red flags that triggered it.

## Architecture

```text
┌──────────────────────────────┐
│ 1. Synthetic Data Generation │  10,000 claims, 9 injected fraud patterns, realistic noise
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ 2. Feature Engineering       │  missing-value flags, ratios, red-flag indicators,
│    (leakage-safe)            │  one-hot encoding, log1p + StandardScaler
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ 3. Isolation Forest Training │  unsupervised, train split only, is_fraud never seen
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ 4. Rule-Based Scoring        │  rule_score = number of 6 business red flags triggered
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ 5. Combined Risk Level       │  High = anomaly AND rule_score ≥ 2
│                              │  Medium = only one signal  ·  Low = neither
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ 6. FastAPI Backend           │  /score_claim, /score_batch, /health, reuses src/ pipeline
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ 7. Frontend Test UI          │  single-file HTML form with sample claims and result card
└──────────────────────────────┘
```

## Dataset

A **synthetic but realistic** dataset built to behave like real claims data, so model results are meaningful rather than trivially separable.

| Property | Value |
|---|---|
| Claims | 10,000 |
| Fraud rate | 5.00% (500 claims) |
| Claim types | Health / Auto / Property (≈ 40 / 35 / 25%) |
| Claimants | 7,858, of whom 17.8% file more than one claim |
| Providers | 200 hospitals, garages and contractors (10 colluding) |
| Amounts | Log-normal per claim type (INR), claimant-consistent |
| Missing provider codes | 135 claims (1.35%), kept as an explicit `UNKNOWN` category |

**Key fields:** claim and policy dates, `claim_type`, `claim_amount`, `policy_coverage_limit`, `claimant_avg_past_claim`, `claimant_claim_count_6m`, `hospital_garage_code`, recent address/bank change flags, and days from policy start and from incident to filing.

**The 9 injected fraud patterns.** Most fraud cases carry 2-3 of these at once:

1. **Early policy claim**: filed 15-30 days after the policy was bought
2. **Post-renewal large claim**: a big claim within 30 days of renewal
3. **Claim spike**: 2.5-5× the claimant's historical average
4. **Round numbers**: suspiciously round amounts (e.g. ₹2,90,000)
5. **Near coverage limit**: within 2-5% of the sum insured
6. **Frequency abuse**: 3+ claims in 6 months
7. **Provider collusion**: a small pool of high-risk hospitals/garages
8. **Delayed reporting**: 15-45 days between incident and filing
9. **Recent address/bank change**: shortly before filing (payout diversion)

To keep the problem honest, **~25% of genuine claims also carry one red flag** and some fraud cases trigger only one, so no single rule separates the classes.

## Key Engineering Decisions

> **This is where most of the thinking went.** A model is only as trustworthy as the pipeline around it.

- **🔒 Data leakage prevention**
  - `is_fraud` is **never passed to `model.fit()`**. Features and labels are saved to **separate files**, joined on `claim_id` only for evaluation.
  - `provider_fraud_rate` is built from labels, so it is computed with **out-of-fold encoding** (each claim's value comes from the other folds) and kept out of the model inputs. The naive in-sample version correlates **+0.323** with fraud versus **+0.275** out-of-fold; that gap is the leakage the out-of-fold method removes.
  - Claimant history features use **strictly earlier claims only**, so nothing looks ahead in time.

- **🎯 Why `contamination=0.05`**
  It matches the known fraud rate, so the number of flagged claims is about the number of real frauds (the threshold is learned on train: 5.00% flagged on train, 4.80% on test), which makes precision and recall directly comparable. It is a business lever rather than a learned parameter: in production it would be set by **investigator capacity**, trading recall against review workload.

- **📊 Why both ROC-AUC and PR-AUC**
  The Isolation Forest scores a respectable **ROC-AUC of 0.862**, yet only **42.7%** of the claims it flags are fraud. With 5% positives, ROC-AUC flatters the model. **PR-AUC (0.361)** is the more honest number, measured against a random baseline of 0.05 instead of 0.5.

- **🤝 Why hybrid rules + ML instead of ML alone**
  - On this data the **rules alone rank fraud better** than the Isolation Forest (ROC-AUC **0.918** vs **0.862**). That's expected: the rules encode the same fraud patterns that were injected into the data, while the forest has to discover "unusual" with no labels.
  - The two signals are **complementary rather than redundant**. Requiring both for "High" raises precision to **52.9%**, above rules alone (46.1%) or the model alone (42.7%).
  - Using either signal ("High" + "Medium") lifts recall to **70%**. That produces a tiered review queue instead of a single yes/no flag.
  - The rules give investigators a **human-readable reason** for every flag. The label-free model is the component that can surface unusual claims no rule was written for.

- **⚙️ Production-minded details**
  - `log1p` is applied to heavy-tailed money columns before scaling.
  - One-hot categories are fixed, so a single incoming claim produces identical columns.
  - The API **imports the exact training pipeline** from `src/` instead of re-implementing it, so training and serving can't drift apart.

## Results

All metrics are on a **held-out, stratified 20% test set**: 2,000 claims, 100 of them fraud (5.00%). The model was trained on the other 8,000 claims without labels.

### Ranking quality (threshold-free)

| Metric | Isolation Forest | Rule score (baseline) | Random |
|---|---|---|---|
| ROC-AUC | **0.862** | **0.918** | 0.500 |
| PR-AUC | **0.361** (7.2× random) | n/a | 0.050 |

### Flagging strategies compared

| Strategy | Claims flagged | Precision | Recall | F1 |
|---|---|---|---|---|
| Isolation Forest @ contamination = 0.05 | 96 (4.8%) | 0.427 | 0.410 | 0.418 |
| Rules only (rule_score ≥ 2) | 141 (7.0%) | 0.461 | 0.650 | **0.539** |
| **High risk bucket** (model AND rules) | **68 (3.4%)** | **0.529** | 0.360 | 0.429 |
| High + Medium (model OR rules) | 169 (8.5%) | 0.414 | **0.700** | 0.520 |

### Risk buckets

| Risk level | Claims | Fraud | Fraud rate | Share of all fraud | Lift vs 5% base rate |
|---|---|---|---|---|---|
| 🔴 **High** | 68 | 36 | **52.9%** | 36.0% | **10.6×** |
| 🟠 **Medium** | 101 | 34 | 33.7% | 34.0% | 6.7× |
| ⚪ **Low** | 1,831 | 30 | 1.6% | 30.0% | 0.3× |

### Key findings

- **High bucket = 10.6× lift.** Reviewing just **3.4%** of claims catches **36%** of all fraud, and more than half of those reviews hit real fraud.
- **Wider net: 70% recall.** Reviewing High + Medium (**8.5%** of claims) catches 70 of the 100 fraud cases.
- **Rules beat the model alone, and the hybrid beats both on precision.** This is an honest result rather than a failure: the rules mirror the known fraud patterns, and the forest adds a second, independent opinion that sharpens the top of the queue.
- **Limitation:** 30% of fraud still lands in **Low**. These are mostly the deliberately weak, single-flag cases that look like genuine claims.

### ROC Curve
![ROC curve](reports/figures/roc_curve.png)
*Both methods rank fraud well above random: the rule score reaches ROC-AUC 0.918 and the Isolation Forest 0.862. The marker shows the model's actual operating point at the 5% contamination threshold.*

### Precision-Recall Curve
![Precision-recall curve](reports/figures/precision_recall_curve.png)
*The Isolation Forest's PR-AUC of 0.361 is 7.2× the 5% random baseline. The diamond marks the High risk bucket at 52.9% precision and 36% recall, the highest precision of any strategy.*

### Confusion Matrix
![Confusion matrix](reports/figures/confusion_matrix.png)
*At the 5% threshold the model catches 41 of 100 fraud cases. It sends 55 genuine claims (2.9% of all genuine claims) to review and misses 59 fraud cases.*

### Risk Bucket Analysis
![Risk bucket analysis](reports/figures/risk_bucket_analysis.png)
*Only 68 of 2,000 claims land in High, but 52.9% of them are fraud. Low holds 1,831 claims at a 1.6% fraud rate, so investigators can safely deprioritise most of the book.*

### Anomaly Score Distribution
![Anomaly score distribution](reports/figures/anomaly_score_distribution.png)
*Fraud claims skew toward higher anomaly scores, but the two distributions overlap substantially. That is why the model alone reaches only 42.7% precision, and why pairing it with rules helps.*

### Feature Correlation with Fraud
![Feature correlation](reports/figures/feature_correlation.png)
*The strongest signals are the near-coverage-limit flag (+0.316; 40.2% of flagged claims are fraud) and round claim amounts (+0.303; 29.8%). Calendar features are close to zero. This is analysis only; these correlations were never used for training.*

## Tech Stack

| Area | Tools |
|---|---|
| Language | Python |
| Data | pandas, numpy |
| Modeling | scikit-learn (IsolationForest, StandardScaler, StratifiedKFold) |
| Visualization | matplotlib, seaborn |
| API | FastAPI, uvicorn, Pydantic |
| Frontend | Plain HTML/CSS/JavaScript (no build step) |

## Project Structure

```text
insurance-fraud-detection/
├── api/
│   └── main.py                      # FastAPI app: /score_claim, /score_batch, /health
├── data/
│   ├── raw/                         # insurance_claims.csv, provider_master.csv
│   └── processed/                   # insurance_claims_features.csv, insurance_claims_labels.csv
├── models/                          # isolation_forest_model.pkl, scaler.pkl, feature_columns.json
├── notebooks/                       # exploration and analysis
├── charts/
│   └── figures/                     # evaluation charts (PNG)
├── source_code/
│   ├── generate_data.py             # synthetic dataset with injected fraud patterns
│   ├── feature_engineering.py       # leakage-safe preprocessing + scaler
│   ├── train_model.py               # Isolation Forest, rule score, evaluation
│   └── visualize_results.py         # portfolio-ready charts
├── frontend/
│   └── index.html                   # manual test UI for the API
├── requirements.txt
└── README.md
```

## How to Run

**1. Clone and install**

```bash
git clone https://github.com/<your-username>/insurance-fraud-detection.git
cd insurance-fraud-detection
pip install -r requirements.txt
```

**2. (Optional) Regenerate the synthetic dataset.** Only needed if `data/raw/` is empty.

```bash
python src/generate_data.py
```

**3. Build features**

```bash
python src/feature_engineering.py
```

**4. Train and evaluate the model**

```bash
python src/train_model.py
```

**5. (Optional) Generate the charts**

```bash
python src/visualize_results.py
```

**6. Start the API**

```bash
uvicorn api.main:app --reload
```

Interactive docs are then available at http://127.0.0.1:8000/docs.

**7. Open the test UI.** Open `frontend/index.html` in a browser, click **Load Sample: Suspicious Claim**, then **Score This Claim**.

> **Note:** the browser calls the API from a different origin, so the API needs CORS enabled (`CORSMiddleware` in `api/main.py`). If the UI shows "API unreachable" while the server is running, that's the cause.

## API Usage Example

**Endpoints:** `POST /score_claim` · `POST /score_batch` · `GET /health` · `GET /`

### curl

```bash
curl -X POST http://127.0.0.1:8000/score_claim \
  -H "Content-Type: application/json" \
  -d '{
    "claim_id": "CLM_TEST_SUS_001",
    "claim_type": "health",
    "claim_amount": 290000,
    "policy_coverage_limit": 300000,
    "claimant_avg_past_claim": null,
    "claimant_claim_count_6m": 1,
    "hospital_garage_code": "HSP0042",
    "policy_renewal_date": null,
    "claim_filed_date": "2026-09-10",
    "days_policy_to_claim": 21,
    "days_incident_to_claim": 17,
    "claimant_address_changed_recently": false,
    "claimant_bank_details_changed_recently": true,
    "is_weekend_filed": false
  }'
```

### Python

```python
import requests

claim = {
    "claim_id": "CLM_TEST_SUS_001",
    "claim_type": "health",
    "claim_amount": 290000,
    "policy_coverage_limit": 300000,
    "claimant_avg_past_claim": None,         # first-time claimant
    "claimant_claim_count_6m": 1,
    "hospital_garage_code": "HSP0042",
    "policy_renewal_date": None,             # never renewed
    "claim_filed_date": "2026-09-10",
    "days_policy_to_claim": 21,
    "days_incident_to_claim": 17,
    "claimant_address_changed_recently": False,
    "claimant_bank_details_changed_recently": True,
    "is_weekend_filed": False,
}

response = requests.post("http://127.0.0.1:8000/score_claim", json=claim, timeout=10)
response.raise_for_status()
print(response.json())
```

### Example response

*(Illustrative: the exact `anomaly_score`, and therefore the risk level, depend on your trained model.)*

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

Invalid input (e.g. an unsupported `claim_type` or a missing field) returns a **400** listing each problem field; if the model isn't loaded, scoring returns **503**.

## Future Improvements

- **Supervised model comparison**: rules already beat the unsupervised model (ROC-AUC 0.918 vs 0.862), so the next step is benchmarking XGBoost/LightGBM with out-of-fold provider encoding.
- **Feature selection for the Isolation Forest**: drop near-zero-signal inputs (`claim_month`, `claim_day_of_week`, `has_been_renewed`), which add noise to a distance-free but split-based detector.
- **Catch the weak cases**: 30% of fraud falls in Low; weighted rule scores or a supervised layer could recover single-flag fraud.
- **SHAP explainability**: per-claim feature attributions alongside the rule flags.
- **Database integration**: compute claimant-history features (`claimant_avg_past_claim`, `claimant_claim_count_6m`) server-side from a claims database instead of trusting the caller.
- **Stricter evaluation**: fit the scaler inside a scikit-learn `Pipeline` on the train split only, and add a time-based split.
- **API security**: API-key or OAuth2 authentication, rate limiting, request logging.
- **Deployment & MLOps**: Docker image, cloud deployment (AWS / GCP / Azure), CI tests, model versioning and drift monitoring.
- **Threshold tuning**: set `contamination` and the rule threshold from investigator capacity and the cost of missed fraud.
