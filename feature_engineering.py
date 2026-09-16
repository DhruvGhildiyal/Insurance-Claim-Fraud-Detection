"""
Feature engineering & preprocessing pipeline for insurance claim fraud detection.

Pipeline (one function per step, so each step is easy to explain):
    1. load_data()                  -> read claims + provider master, parse dates
    2. handle_missing_values()      -> missing-indicator flags, then sensible fills
    3. merge_provider_info()        -> provider name / city / type (readability + data-quality check)
    4. engineer_features()          -> ratios, red-flag indicators, calendar features
    5. encode_categoricals()        -> one-hot claim_type with a FIXED category list
    6. scale_numeric_features()     -> log1p on skewed money columns, then StandardScaler
    7. provider fraud rate          -> label-based, so computed out-of-fold for analysis only
    8. print_summary()              -> feature list, shapes, correlation with is_fraud

Outputs (same folder as this script)
    insurance_claims_features.csv   claim_id + model input features (NO label, NO label-derived columns)
    insurance_claims_labels.csv     claim_id + is_fraud + provider info + provider_fraud_rate (evaluation only)
    scaler.pkl                      fitted StandardScaler (joblib)
    feature_columns.json            exact column order / log columns the backend must reproduce

Leakage rules this script follows
    - is_fraud never enters the feature matrix; it is written to a separate file.
    - provider_fraud_rate is built from labels, so it is NOT a model feature here. It is
      computed out-of-fold for analysis, and fit_provider_fraud_rate() /
      apply_provider_fraud_rate() are provided for later: fit on the TRAIN split only.
    - engineer_features() uses only the claim's own row, so the backend can call the
      exact same function on a single incoming claim.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# =============================================================================
# Configuration
# =============================================================================
# Project root on disk. This script lives in <PROJECT_ROOT>/src/, so BASE_DIR.parent
# gives the project root: D:\projects1\insurance-fraud-detection
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Inputs (place these two files in data/raw/)
CLAIMS_CSV = DATA_RAW_DIR / "insurance_claims.csv"
PROVIDER_CSV = DATA_RAW_DIR / "provider_master.csv"

# Outputs
FEATURES_CSV = DATA_PROCESSED_DIR / "insurance_claims_features.csv"
LABELS_CSV = DATA_PROCESSED_DIR / "insurance_claims_labels.csv"
SCALER_PKL = MODELS_DIR / "scaler.pkl"
FEATURE_COLUMNS_JSON = MODELS_DIR / "feature_columns.json"

# Make sure the output folders exist before we try to save anything into them
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
ID_COL = "claim_id"
LABEL_COL = "is_fraud"
DATE_COLS = ["policy_start_date", "policy_renewal_date", "incident_date", "claim_filed_date"]

# Business-rule thresholds for the indicator features
EARLY_POLICY_DAYS = 30
NEAR_LIMIT_RATIO = 0.90
HIGH_FREQUENCY_COUNT = 3
DELAYED_REPORTING_DAYS = 15
ROUND_UNIT = 1_000
NEUTRAL_RATIO = 1.0              # claim_to_avg_ratio for first-time claimants ("same as usual")
PROVIDER_RATE_SMOOTHING = 20     # pseudo-claims pulling small providers toward the global rate

# Fixed list so a single claim at inference time still produces all one-hot columns
CLAIM_TYPES = ["auto", "health", "property"]

# Columns the raw input must contain (training CSV or a backend request)
REQUIRED_INPUT_COLUMNS = [
    ID_COL, "policy_renewal_date", "claim_filed_date", "claim_type", "claim_amount",
    "policy_coverage_limit", "claimant_avg_past_claim", "claimant_claim_count_6m",
    "hospital_garage_code", "claimant_address_changed_recently",
    "claimant_bank_details_changed_recently", "is_weekend_filed",
    "days_policy_to_claim", "days_incident_to_claim",
]

# Continuous / count features -> StandardScaler
NUMERIC_FEATURES = [
    "claim_amount", "policy_coverage_limit", "claimant_avg_past_claim",
    "claimant_claim_count_6m", "days_policy_to_claim", "days_incident_to_claim",
    "claim_to_avg_ratio", "claim_to_limit_ratio", "claim_month", "claim_day_of_week",
]
# Money amounts and the past-average ratio are log-normal / heavy-tailed. Without log1p a
# handful of huge claims dominate the z-scores (and any distance-based anomaly model).
LOG_TRANSFORM_COLS = ["claim_amount", "policy_coverage_limit",
                      "claimant_avg_past_claim", "claim_to_avg_ratio"]
# 0/1 flags -> left as-is (already on a comparable scale, and stay interpretable)
BINARY_FEATURES = [
    "is_first_time_claimant", "has_been_renewed", "is_provider_unknown", "is_round_number",
    "is_early_policy_claim", "is_near_limit", "is_high_frequency_claimant",
    "is_delayed_reporting", "any_recent_change", "claimant_address_changed_recently",
    "claimant_bank_details_changed_recently", "is_weekend_filed",
]
ONEHOT_FEATURES = [f"claim_type_{t}" for t in CLAIM_TYPES]
MODEL_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + ONEHOT_FEATURES

# The features created in this script (used for the correlation check)
ENGINEERED_FEATURES = [
    "is_first_time_claimant", "has_been_renewed", "is_provider_unknown",
    "claim_to_avg_ratio", "claim_to_limit_ratio", "is_round_number", "claim_month",
    "claim_day_of_week", "is_early_policy_claim", "is_near_limit",
    "is_high_frequency_claimant", "is_delayed_reporting", "any_recent_change",
]

assert LABEL_COL not in MODEL_FEATURES, "the label must never be a model input"


# =============================================================================
# Helpers
# =============================================================================
def to_flag(series):
    """Robust 0/1 conversion: works for bool dtype, 0/1 ints and 'True'/'false' strings (JSON)."""
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "1.0", "yes"]).astype(int)


def validate_input(df):
    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")


# =============================================================================
# Step 1: load
# =============================================================================
def load_data(claims_path=CLAIMS_CSV, provider_path=PROVIDER_CSV):
    claims = pd.read_csv(claims_path, parse_dates=DATE_COLS)
    providers = pd.read_csv(provider_path)
    validate_input(claims)
    return claims, providers


# =============================================================================
# Step 2: missing values
# =============================================================================
def handle_missing_values(df):
    """Record WHY a value is missing as a flag before filling it - the fact that it was
    missing is often more informative than the filled value."""
    out = df.copy()

    # claimant_avg_past_claim is null when there is no claim history
    out["is_first_time_claimant"] = out["claimant_avg_past_claim"].isna().astype(int)
    out["claimant_avg_past_claim"] = out["claimant_avg_past_claim"].fillna(0.0)

    # policy_renewal_date is null when the claim falls in the policy's first term
    out["has_been_renewed"] = out["policy_renewal_date"].notna().astype(int)

    # Provider not captured on the claim form: keep it as its own explicit category
    out["is_provider_unknown"] = out["hospital_garage_code"].isna().astype(int)
    out["hospital_garage_code"] = out["hospital_garage_code"].fillna("UNKNOWN")
    return out


# =============================================================================
# Step 3: provider master merge
# =============================================================================
def merge_provider_info(df, providers):
    """Attach provider name / city / type.

    provider_master.csv has no risk-category column, so nothing from it is a model
    feature: name and city are identifiers, and provider type is fully determined by
    claim_type. The merge is used for (a) a data-quality check that every code exists
    and matches the claim's line of business and (b) a readable evaluation table.
    """
    lookup = providers.rename(columns={
        "provider_code": "hospital_garage_code",
        "city": "provider_city",
        "claim_type": "provider_claim_type",
    })
    out = df.merge(lookup, on="hospital_garage_code", how="left", validate="many_to_one")

    known = out["hospital_garage_code"] != "UNKNOWN"
    quality = {
        "codes_not_in_master": int((known & out["provider_name"].isna()).sum()),
        "type_mismatches": int((known & out["provider_name"].notna()
                                & (out["provider_claim_type"] != out["claim_type"])).sum()),
    }
    info_cols = ["provider_name", "provider_city", "provider_claim_type"]
    out[info_cols] = out[info_cols].fillna("UNKNOWN")
    return out, quality


# =============================================================================
# Step 4: engineered features (row-level only -> safe for single-claim inference)
# =============================================================================
def engineer_features(df):
    out = df.copy()
    amount = out["claim_amount"].astype(float)
    past_avg = out["claimant_avg_past_claim"].astype(float)
    limit = out["policy_coverage_limit"].astype(float)
    filed = pd.to_datetime(out["claim_filed_date"])

    # How unusual is this claim for THIS claimant? First-time claimants have no baseline,
    # so they get a neutral 1.0 ("in line with their history") instead of inf / NaN.
    out["claim_to_avg_ratio"] = (amount / past_avg.where(past_avg > 0)).fillna(NEUTRAL_RATIO)

    # How much of the cover is being used? Fraudsters tend to max out the policy.
    out["claim_to_limit_ratio"] = (amount / limit.where(limit > 0)).fillna(0.0)

    # Genuine bills are itemised (e.g. 48,317); invented amounts are often round (50,000)
    out["is_round_number"] = (amount % ROUND_UNIT == 0).astype(int)

    # Calendar features from the filing date (day_of_week: 0 = Monday)
    out["claim_month"] = filed.dt.month
    out["claim_day_of_week"] = filed.dt.dayofweek

    # Rule-style indicators mirroring known fraud red flags
    out["is_early_policy_claim"] = (out["days_policy_to_claim"] <= EARLY_POLICY_DAYS).astype(int)
    out["is_near_limit"] = (out["claim_to_limit_ratio"] >= NEAR_LIMIT_RATIO).astype(int)
    out["is_high_frequency_claimant"] = (
        out["claimant_claim_count_6m"] >= HIGH_FREQUENCY_COUNT).astype(int)
    out["is_delayed_reporting"] = (
        out["days_incident_to_claim"] >= DELAYED_REPORTING_DAYS).astype(int)

    # Normalise boolean columns to 0/1, then combine the two KYC-change signals
    for col in ["claimant_address_changed_recently", "claimant_bank_details_changed_recently",
                "is_weekend_filed"]:
        out[col] = to_flag(out[col])
    out["any_recent_change"] = (out["claimant_address_changed_recently"]
                                | out["claimant_bank_details_changed_recently"]).astype(int)
    return out


# =============================================================================
# Step 5: categorical encoding
# =============================================================================
def encode_categoricals(df):
    """One-hot claim_type. Casting to a Categorical with a fixed category list guarantees
    the same three columns in the same order, even for a single 'auto' claim."""
    out = df.copy()
    claim_type = pd.Categorical(out["claim_type"].str.strip().str.lower(), categories=CLAIM_TYPES)
    if pd.isna(claim_type).any():
        bad = sorted(set(out.loc[pd.isna(claim_type), "claim_type"]))
        raise ValueError(f"Unknown claim_type values: {bad}")
    dummies = pd.get_dummies(claim_type, prefix="claim_type", dtype=int)
    dummies.index = out.index
    return pd.concat([out, dummies], axis=1)


# =============================================================================
# Step 6: scaling
# =============================================================================
def scale_numeric_features(df, scaler=None):
    """log1p the heavy-tailed columns, then standardise all numeric features.

    scaler=None  -> fit a new StandardScaler (training)
    scaler=obj   -> reuse a fitted one (backend inference / test split), never refit
    """
    out = df.copy()
    X = out[NUMERIC_FEATURES].astype(float)
    for col in LOG_TRANSFORM_COLS:
        X[col] = np.log1p(X[col])
    if scaler is None:
        scaler = StandardScaler().fit(X)
    out[NUMERIC_FEATURES] = scaler.transform(X)
    return out, scaler


# =============================================================================
# Full reusable pipeline (training AND backend)
# =============================================================================
def preprocess(raw_claims, scaler=None):
    """Raw claims -> (model-ready feature table, scaler, unscaled engineered table).

    The backend loads scaler.pkl and calls preprocess(new_claims_df, scaler=scaler).
    """
    validate_input(raw_claims)
    df = handle_missing_values(raw_claims)
    df = engineer_features(df)
    df = encode_categoricals(df)
    scaled, scaler = scale_numeric_features(df, scaler)
    features = scaled[[ID_COL] + MODEL_FEATURES].reset_index(drop=True)
    return features, scaler, df


# =============================================================================
# Step 7: provider fraud rate (label-based -> handle with care)
# =============================================================================
def fit_provider_fraud_rate(train_df, smoothing=PROVIDER_RATE_SMOOTHING):
    """Smoothed fraud rate per provider, learned from TRAINING rows only.

    rate = (fraud_count + smoothing * global_rate) / (claim_count + smoothing)
    A provider with 3 claims and 1 fraud is not really "33% fraud": smoothing pulls
    small providers toward the global rate, big providers keep their own rate.
    """
    prior = float(train_df[LABEL_COL].mean())
    stats = train_df.groupby("hospital_garage_code")[LABEL_COL].agg(["sum", "count"])
    rates = (stats["sum"] + smoothing * prior) / (stats["count"] + smoothing)
    return {"rates": rates.to_dict(), "prior": prior, "smoothing": smoothing}


def apply_provider_fraud_rate(df, encoding):
    """Providers never seen in training get the global prior."""
    return df["hospital_garage_code"].map(encoding["rates"]).fillna(encoding["prior"]).astype(float)


def out_of_fold_provider_fraud_rate(df, n_splits=5, seed=SEED):
    """Each claim's provider rate is computed from the OTHER folds, so a claim's own label
    never feeds its own feature value. Mimics what a model sees on unseen data."""
    oof = pd.Series(np.nan, index=df.index, dtype=float)
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, valid_idx in folds.split(df, df[LABEL_COL]):
        encoding = fit_provider_fraud_rate(df.iloc[train_idx])
        oof.iloc[valid_idx] = apply_provider_fraud_rate(df.iloc[valid_idx], encoding).to_numpy()
    return oof


def build_label_table(df):
    """Evaluation-only table: label, readable provider info, label-derived provider rate."""
    table = df[[ID_COL, LABEL_COL, "claim_type", "hospital_garage_code",
                "provider_name", "provider_city"]].copy()
    table["provider_fraud_rate"] = out_of_fold_provider_fraud_rate(df).round(6)
    return table.reset_index(drop=True)


# =============================================================================
# Step 8: summary
# =============================================================================
def correlation_report(engineered, labels):
    """Correlation of each engineered feature with is_fraud (unscaled values).

    Pearson = linear relationship; Spearman = rank-based, fairer for skewed ratios.
    For 0/1 flags, 'fraud rate if 1' is the most interpretable number.
    """
    y = labels[LABEL_COL].to_numpy()
    data = engineered[ENGINEERED_FEATURES].reset_index(drop=True).copy()
    data["provider_fraud_rate (out-of-fold)"] = labels["provider_fraud_rate"]
    # Deliberately leaky version, shown only to demonstrate why out-of-fold matters
    data["provider_fraud_rate (in-sample, LEAKY)"] = (
        labels.groupby("hospital_garage_code")[LABEL_COL].transform("mean"))

    rows = []
    for col in data.columns:
        x = data[col].astype(float)
        is_flag = set(x.unique()) <= {0.0, 1.0}
        rows.append({
            "feature": col,
            "pearson": np.corrcoef(x, y)[0, 1] if x.std() > 0 else np.nan,
            "spearman": x.rank().corr(pd.Series(y).rank()),
            "share_of_claims_if_flag": x.mean() if is_flag else np.nan,
            "fraud_rate_if_flag=1": y[x == 1].mean() if is_flag and (x == 1).any() else np.nan,
        })
    report = pd.DataFrame(rows).set_index("feature")
    return report.reindex(report["pearson"].abs().sort_values(ascending=False).index)


def print_summary(features, labels, engineered, quality):
    line = "=" * 96
    X = features[MODEL_FEATURES]
    print(line)
    print("FEATURE ENGINEERING SUMMARY")
    print(line)

    print(f"\nFinal model features ({len(MODEL_FEATURES)})")
    print(f"  numeric, standardised ({len(NUMERIC_FEATURES)}): {', '.join(NUMERIC_FEATURES)}")
    print(f"    log1p applied before scaling to: {', '.join(LOG_TRANSFORM_COLS)}")
    print(f"  binary 0/1 ({len(BINARY_FEATURES)}): {', '.join(BINARY_FEATURES)}")
    print(f"  one-hot ({len(ONEHOT_FEATURES)}): {', '.join(ONEHOT_FEATURES)}")

    print("\nShapes")
    print(f"  {FEATURES_CSV.name:<32} {features.shape}  (claim_id + {X.shape[1]} features)")
    print(f"  model input matrix               {X.shape}")
    print(f"  {LABELS_CSV.name:<32} {labels.shape}  (evaluation only)")
    print(f"  missing values in model matrix   {int(X.isna().sum().sum())}")
    print(f"  label/leaky columns in features  "
          f"{[c for c in features.columns if c in (LABEL_COL, 'provider_fraud_rate')] or 'none'}")

    print("\nProvider merge")
    print(f"  claims with unknown provider     {int(engineered['is_provider_unknown'].sum())}")
    print(f"  codes not found in master        {quality['codes_not_in_master']}")
    print(f"  provider type != claim type      {quality['type_mismatches']}")

    print("\nScaled numeric features (should be mean ~0, std ~1)")
    print(X[NUMERIC_FEATURES].agg(["mean", "std"]).T.round(3).to_string())

    print(f"\nCorrelation with {LABEL_COL} (analysis only - not used for training)")
    print(f"  overall fraud rate: {labels[LABEL_COL].mean():.2%}")
    report = correlation_report(engineered, labels)
    shown = pd.DataFrame(index=report.index)
    for col in ["pearson", "spearman"]:
        shown[col] = report[col].map("{:+.3f}".format)
    for col in ["share_of_claims_if_flag", "fraud_rate_if_flag=1"]:
        shown[col] = report[col].map(lambda v: "-" if pd.isna(v) else f"{v:.1%}")
    print(shown.to_string())
    print("\n  Note: the in-sample provider rate looks stronger only because each claim's own")
    print("  label is baked into its value. Use fit_provider_fraud_rate() on the train split.")
    print(line)


# =============================================================================
# Main
# =============================================================================
def main():
    claims, providers = load_data()

    # Model features: the same function the backend will call
    features, scaler, engineered = preprocess(claims)

    # Evaluation-only artefacts: provider info + label + label-derived provider rate
    with_providers, quality = merge_provider_info(handle_missing_values(claims), providers)
    labels = build_label_table(with_providers)
    assert (labels[ID_COL].to_numpy() == features[ID_COL].to_numpy()).all()

    features.to_csv(FEATURES_CSV, index=False, float_format="%.6f")
    labels.to_csv(LABELS_CSV, index=False)
    joblib.dump(scaler, SCALER_PKL)
    FEATURE_COLUMNS_JSON.write_text(json.dumps({
        "id_column": ID_COL,
        "model_features": MODEL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "log_transform_columns": LOG_TRANSFORM_COLS,
        "binary_features": BINARY_FEATURES,
        "onehot_features": ONEHOT_FEATURES,
        "claim_types": CLAIM_TYPES,
    }, indent=2))

    print(f"Saved {FEATURES_CSV.name}, {LABELS_CSV.name}, {SCALER_PKL.name}, "
          f"{FEATURE_COLUMNS_JSON.name}\n")
    print_summary(features, labels, engineered, quality)


if __name__ == "__main__":
    main()