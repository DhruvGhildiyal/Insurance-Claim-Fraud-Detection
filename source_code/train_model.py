"""
Train an Isolation Forest fraud detector and combine it with a transparent rule
score into a 3-level risk rating (Low / Medium / High).

Pipeline (one function per step):
    1. load_data()               -> features (X) + labels (evaluation only), merged on claim_id
    2. split_data()              -> 80/20 train/test split, stratified by is_fraud
    3. train_isolation_forest()  -> fit on TRAIN features only; no label is ever passed
    4. score_claims()            -> anomaly score, IF flag, rule_score, combined_risk_level
    5. evaluate()                -> metrics on the TEST set only
    6. print_summary()           -> interview-ready report
    7. save_model()              -> models/isolation_forest_model.pkl

Run from the project root (python src/train_model.py) or from inside src/.
All paths are built from PROJECT_ROOT, so the working directory does not matter.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

# =============================================================================
# Paths (same PROJECT_ROOT pattern as feature_engineering.py)
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent      # src/ -> project root
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

FEATURES_CSV = PROCESSED_DIR / "insurance_claims_features.csv"
LABELS_CSV = PROCESSED_DIR / "insurance_claims_labels.csv"
MODEL_PKL = MODELS_DIR / "isolation_forest_model.pkl"

# =============================================================================
# Configuration
# =============================================================================
ID_COL = "claim_id"
LABEL_COL = "is_fraud"
LEAKY_COLUMNS = {LABEL_COL, "provider_fraud_rate"}   # label or label-derived: never model inputs

RANDOM_STATE = 42
TEST_SIZE = 0.20

# --- Isolation Forest hyperparameters ----------------------------------------
# contamination = expected share of anomalies in the training data.
#   Isolation Forest is unsupervised: contamination does NOT change how the trees are
#   built. It only sets the cut-off on the anomaly score (model.offset_) so that ~5% of
#   TRAIN claims are flagged. 0.05 matches the known portfolio fraud rate, so the number
#   of flagged claims is roughly the number of real frauds, which keeps precision and
#   recall directly comparable.
#   Trade-off: a higher value catches more fraud (recall up) but sends more genuine
#   claims to investigators (precision down); a lower value does the opposite. In
#   production this is usually set by investigation capacity (how many claims the fraud
#   team can review per week), not only by the historical fraud rate.
#   Caveat: using the known fraud rate is a mild use of label knowledge. It is a business
#   prior, not something the model learns from is_fraud.
CONTAMINATION = 0.05

# n_estimators = number of isolation trees.
#   Each tree isolates points with random splits; the anomaly score is the average path
#   length across trees. More trees give a more stable, less noisy ranking, but the
#   benefit flattens after ~100-200 trees while training time, scoring latency and model
#   size keep growing linearly. 200 is a safe balance between stability and cost.
N_ESTIMATORS = 200

# max_samples = rows used to build each tree ('auto' = min(256, n_rows)).
#   Small subsamples are deliberate: they reduce "masking" (a dense group of anomalies
#   hiding each other) and "swamping" (normal points next to anomalies being flagged).
#   Larger values rarely help and make trees deeper and slower. Kept at the default.
MAX_SAMPLES = "auto"

# --- Rule-based score ----------------------------------------------------------
# Binary red-flag features created in feature_engineering.py (still 0/1, not scaled)
RULE_COLUMNS = [
    "is_early_policy_claim", "is_near_limit", "is_high_frequency_claimant",
    "is_delayed_reporting", "any_recent_change", "is_round_number",
]
RULE_FLAG_THRESHOLD = 2          # rule_score >= 2 counts as "rules flag this claim"
RISK_LEVELS = ["High", "Medium", "Low"]


# =============================================================================
# Step 1: load
# =============================================================================
def load_data():
    """Load X and labels, merge on claim_id. Returns (data, feature_cols).

    feature_cols comes ONLY from the features file, so nothing from the labels file
    (is_fraud, provider_fraud_rate) can reach the model.
    """
    features = pd.read_csv(FEATURES_CSV)
    labels = pd.read_csv(LABELS_CSV, usecols=[ID_COL, LABEL_COL])

    feature_cols = [c for c in features.columns if c != ID_COL]
    leaks = LEAKY_COLUMNS & set(feature_cols)
    if leaks:
        raise ValueError(f"Label/label-derived columns found in features file: {sorted(leaks)}")
    missing_rules = [c for c in RULE_COLUMNS if c not in feature_cols]
    if missing_rules:
        raise ValueError(f"Rule columns missing from features file: {missing_rules}")

    # Merge is only so the label travels with each row through the split for evaluation
    data = features.merge(labels, on=ID_COL, how="inner", validate="one_to_one")
    if len(data) != len(features):
        raise ValueError(f"{len(features) - len(data)} claims have no label in {LABELS_CSV.name}")
    return data, feature_cols


# =============================================================================
# Step 2: split
# =============================================================================
def split_data(data):
    """80/20 split, stratified so both sets keep the ~5% fraud rate (important with
    only ~500 fraud cases: an unstratified split could leave the test set short of fraud)."""
    train, test = train_test_split(
        data, test_size=TEST_SIZE, stratify=data[LABEL_COL], random_state=RANDOM_STATE)
    return train.reset_index(drop=True), test.reset_index(drop=True)


# =============================================================================
# Step 3: train
# =============================================================================
def train_isolation_forest(X_train):
    """Fit on train features only. Isolation Forest takes no y; is_fraud is not passed."""
    assert LABEL_COL not in X_train.columns, "is_fraud must never be a model input"
    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        max_samples=MAX_SAMPLES,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train)
    return model


# =============================================================================
# Step 4: scoring (label-free, so the backend can reuse it)
# =============================================================================
def compute_rule_score(df):
    """Number of red flags raised (0-6)."""
    flags = df[RULE_COLUMNS]
    if not flags.isin([0, 1]).all().all():
        raise ValueError("Rule columns must be 0/1 (were they scaled by mistake?)")
    return flags.astype(int).sum(axis=1)


def assign_risk_level(if_flag, rule_flag):
    """High = both signals agree, Medium = exactly one signal, Low = neither."""
    if_flag, rule_flag = np.asarray(if_flag, bool), np.asarray(rule_flag, bool)
    return np.select([if_flag & rule_flag, if_flag | rule_flag], ["High", "Medium"], default="Low")


def score_claims(model, X, claim_ids):
    """Return one row per claim with model and rule outputs. X must not contain labels."""
    assert LABEL_COL not in X.columns
    scored = pd.DataFrame({ID_COL: np.asarray(claim_ids)})

    # score_samples: higher = MORE normal. Negate so higher = more anomalous (needed for ROC-AUC).
    scored["anomaly_score"] = -model.score_samples(X)

    # predict: -1 = anomaly, 1 = normal, using the cut-off set by contamination on TRAIN
    scored["if_anomaly"] = (model.predict(X) == -1).astype(int)

    scored["rule_score"] = compute_rule_score(X).to_numpy()
    scored["rule_flag"] = (scored["rule_score"] >= RULE_FLAG_THRESHOLD).astype(int)
    scored["combined_risk_level"] = assign_risk_level(scored["if_anomaly"], scored["rule_flag"])
    return scored


# =============================================================================
# Step 5: evaluation (TEST set only)
# =============================================================================
def flag_metrics(y_true, flagged):
    """Precision / recall / F1 for any 0/1 flagging strategy."""
    flagged = np.asarray(flagged).astype(int)
    return {
        "flagged": int(flagged.sum()),
        "flag_rate": flagged.mean(),
        "precision": precision_score(y_true, flagged, zero_division=0),
        "recall": recall_score(y_true, flagged, zero_division=0),
        "f1": f1_score(y_true, flagged, zero_division=0),
    }


def evaluate(scored, y_true):
    y = np.asarray(y_true).astype(int)
    level = scored["combined_risk_level"]

    # Threshold-free ranking quality. PR-AUC matters more than ROC-AUC at 5% fraud,
    # because ROC-AUC can look good while precision is still low.
    ranking = {
        "if_roc_auc": roc_auc_score(y, scored["anomaly_score"]),
        "if_pr_auc": average_precision_score(y, scored["anomaly_score"]),
        "rule_roc_auc": roc_auc_score(y, scored["rule_score"]),
    }

    # Threshold metrics for each flagging strategy
    strategies = pd.DataFrame({
        f"Isolation Forest (contamination={CONTAMINATION})": flag_metrics(y, scored["if_anomaly"]),
        f"Rules only (rule_score >= {RULE_FLAG_THRESHOLD})": flag_metrics(y, scored["rule_flag"]),
        "High risk bucket (IF AND rules)": flag_metrics(y, level == "High"),
        "High + Medium (IF OR rules)": flag_metrics(y, level != "Low"),
    }).T

    # Fraud rate and fraud captured per risk bucket
    buckets = (pd.DataFrame({"combined_risk_level": level.to_numpy(), LABEL_COL: y})
               .groupby("combined_risk_level")[LABEL_COL]
               .agg(claims="size", fraud="sum")
               .reindex(RISK_LEVELS, fill_value=0))
    buckets["fraud_rate"] = buckets["fraud"] / buckets["claims"].replace(0, np.nan)
    buckets["share_of_all_fraud"] = buckets["fraud"] / y.sum()

    return {
        "n_test": len(y),
        "n_fraud": int(y.sum()),
        "base_rate": y.mean(),
        "ranking": ranking,
        "confusion": confusion_matrix(y, scored["if_anomaly"], labels=[0, 1]),
        "strategies": strategies,
        "buckets": buckets,
    }


# =============================================================================
# Step 6: summary
# =============================================================================
def pct(value):
    return "-" if pd.isna(value) else f"{value:.1%}"


def print_summary(results, n_train, train_flag_rate, test_flag_rate, n_features):
    line = "=" * 84
    base = results["base_rate"]
    r = results["ranking"]
    s = results["strategies"]

    print(line)
    print("INSURANCE FRAUD - ISOLATION FOREST + RULES | EVALUATION ON HELD-OUT TEST SET")
    print(line)

    print("\n1. Setup")
    print(f"  Train / test claims        : {n_train:,} / {results['n_test']:,}  (stratified 80/20)")
    print(f"  Fraud in test set          : {results['n_fraud']:,}  ({base:.2%})")
    print(f"  Model inputs               : {n_features} features (is_fraud never passed to fit)")
    print(f"  IsolationForest            : n_estimators={N_ESTIMATORS}, "
          f"contamination={CONTAMINATION}, max_samples={MAX_SAMPLES!r}")
    print(f"  Anomaly flag rate          : train {train_flag_rate:.2%} | test {test_flag_rate:.2%}")

    print("\n2. Ranking quality (no threshold)")
    print(f"  ROC-AUC  IF anomaly score  : {r['if_roc_auc']:.3f}   (0.5 = random, 1.0 = perfect)")
    print(f"  PR-AUC   IF anomaly score  : {r['if_pr_auc']:.3f}   (random = base rate {base:.3f})")
    print(f"  ROC-AUC  rule_score        : {r['rule_roc_auc']:.3f}   (simple baseline)")

    print(f"\n3. Isolation Forest at the contamination threshold ({CONTAMINATION:.0%})")
    iso = s.iloc[0]
    print(f"  Precision {iso['precision']:.3f} | Recall {iso['recall']:.3f} | F1 {iso['f1']:.3f}")
    tn, fp, fn, tp = results["confusion"].ravel()
    print("\n  Confusion matrix            Pred. normal   Pred. anomaly")
    print(f"    Actual genuine            {tn:>12,}   {fp:>13,}   <- FP: genuine claims sent to review")
    print(f"    Actual fraud              {fn:>12,}   {tp:>13,}   <- FN: fraud missed")

    print("\n4. Flagging strategies compared")
    shown = pd.DataFrame(index=s.index)
    shown["flagged"] = s["flagged"].astype(int).map("{:,}".format)
    shown["% of claims"] = s["flag_rate"].map(pct)
    for col in ["precision", "recall", "f1"]:
        shown[col] = s[col].map("{:.3f}".format)
    print(shown.to_string())

    print("\n5. Combined risk buckets")
    b = results["buckets"]
    shown_b = pd.DataFrame({
        "claims": b["claims"].map("{:,}".format),
        "fraud": b["fraud"].map("{:,}".format),
        "fraud rate (precision)": b["fraud_rate"].map(pct),
        "share of all fraud (recall)": b["share_of_all_fraud"].map(pct),
    })
    print(shown_b.to_string())

    high = s.loc["High risk bucket (IF AND rules)"]
    either = s.loc["High + Medium (IF OR rules)"]
    lift = high["precision"] / base if base > 0 else float("nan")
    print("\n6. Key takeaways")
    print(f"  - A random claim is fraud {base:.1%} of the time; a 'High' claim is fraud "
          f"{high['precision']:.1%} of the time ({lift:.1f}x lift).")
    print(f"  - 'High' alone catches {high['recall']:.1%} of test fraud by reviewing "
          f"{high['flag_rate']:.1%} of claims.")
    print(f"  - 'High' + 'Medium' catches {either['recall']:.1%} of fraud but means reviewing "
          f"{either['flag_rate']:.1%} of claims: the precision/recall trade-off.")
    print("  - The model never saw a label; is_fraud was used only to measure these results.")
    print(line)


# =============================================================================
# Step 7: save
# =============================================================================
def save_model(model):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PKL)
    print(f"\nSaved model -> {MODEL_PKL}")


# =============================================================================
# Main
# =============================================================================
def main():
    data, feature_cols = load_data()
    train, test = split_data(data)

    # Only feature columns go to the model; the label stays behind in train/test
    X_train, X_test = train[feature_cols], test[feature_cols]
    model = train_isolation_forest(X_train)

    train_scored = score_claims(model, X_train, train[ID_COL])   # sanity check: ~5% flagged
    test_scored = score_claims(model, X_test, test[ID_COL])

    # First and only place the label is used: evaluation on the test set
    results = evaluate(test_scored, test[LABEL_COL])
    print_summary(results, len(train), train_scored["if_anomaly"].mean(),
                  test_scored["if_anomaly"].mean(), len(feature_cols))
    save_model(model)


if __name__ == "__main__":
    main()
