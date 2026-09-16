"""
Portfolio-ready charts for the Isolation Forest + rules fraud model.

Reuses train_model.py end to end (load_data -> split_data -> score_claims -> evaluate)
and scores with the SAVED model in models/isolation_forest_model.pkl, so every number
on the charts matches the summary train_model.py printed.

Charts (PNG, 150 dpi) -> reports/figures/
    a) roc_curve.png                   IF anomaly score vs rule_score
    b) precision_recall_curve.png      IF anomaly score vs the 5% base rate
    c) confusion_matrix.png            IF at the contamination threshold
    d) risk_bucket_analysis.png        claim volume vs fraud rate per risk bucket
    e) anomaly_score_distribution.png  fraud vs genuine score distributions
    f) feature_correlation.png         top 10 engineered features vs is_fraud

Run from the project root (python src/visualize_results.py) or from inside src/.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # write files only; no display window needed
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import precision_recall_curve, roc_curve

# Make "from train_model import ..." work wherever the script is launched from
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from train_model import (  # noqa: E402
    CONTAMINATION, ID_COL, LABEL_COL, LABELS_CSV, MODEL_PKL, PROJECT_ROOT, RISK_LEVELS,
    RULE_FLAG_THRESHOLD, evaluate, load_data, score_claims, split_data,
)

# =============================================================================
# Paths & style
# =============================================================================
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
DPI = 150
TOP_N_FEATURES = 10

# Colorblind-safe palette (Okabe-Ito style): blue / orange / vermillion / purple / grey.
# No pure red-vs-green pairs anywhere.
PALETTE = sns.color_palette("colorblind")
COLORS = {
    "if": PALETTE[0],            # blue    - Isolation Forest
    "rules": PALETTE[4],         # purple  - rule score
    "reference": "#7F7F7F",      # grey    - random / base-rate reference lines
    "genuine": PALETTE[0],       # blue
    "fraud": PALETTE[1],         # orange
    "positive": PALETTE[3],      # vermillion - higher value -> more fraud
    "negative": PALETTE[0],      # blue       - higher value -> less fraud
    "high_bucket": PALETTE[3],
}
BUCKET_COLORS = {"High": PALETTE[3], "Medium": PALETTE[1], "Low": PALETTE[7]}
SUBTITLE_COLOR = "#555555"

# Engineered features from feature_engineering.py (used by the fallback correlation path)
ENGINEERED_FEATURES = [
    "is_first_time_claimant", "has_been_renewed", "is_provider_unknown",
    "claim_to_avg_ratio", "claim_to_limit_ratio", "is_round_number", "claim_month",
    "claim_day_of_week", "is_early_policy_claim", "is_near_limit",
    "is_high_frequency_claimant", "is_delayed_reporting", "any_recent_change",
]


def set_style():
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.05)
    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "savefig.facecolor": "white",
    })


def add_titles(fig, ax, title, subtitle):
    """Bold headline for the figure + a smaller grey context line on the axes."""
    fig.suptitle(title, fontsize=14, fontweight="bold")
    ax.set_title(subtitle, fontsize=10, color=SUBTITLE_COLOR, pad=8)


def save_figure(fig, filename):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


# =============================================================================
# Reproduce the exact results from train_model.py
# =============================================================================
def load_trained_model(feature_cols):
    """Load the saved model and confirm it was trained on the same feature columns."""
    if not MODEL_PKL.exists():
        raise FileNotFoundError(f"{MODEL_PKL} not found - run src/train_model.py first.")
    model = joblib.load(MODEL_PKL)
    trained_on = list(getattr(model, "feature_names_in_", feature_cols))
    if trained_on != list(feature_cols):
        raise ValueError("Saved model was trained on different feature columns - "
                         "re-run src/train_model.py.")
    return model


def reproduce_results():
    """Same load -> split -> score -> evaluate path as train_model.py.

    Uses the SAVED model instead of retraining, so the charts always describe the
    model that ships to the backend.
    """
    data, feature_cols = load_data()
    _, test = split_data(data)                      # same stratified split, random_state=42
    model = load_trained_model(feature_cols)
    scored = score_claims(model, test[feature_cols], test[ID_COL])
    y_test = test[LABEL_COL].to_numpy().astype(int)  # label used for evaluation only
    results = evaluate(scored, y_test)
    return data, model, scored, y_test, results


def operating_points(results):
    """(precision, recall, FPR) for the IF threshold, and precision/recall for 'High'."""
    tn, fp, fn, tp = results["confusion"].ravel()
    iso = {
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
    }
    b = results["buckets"]
    high = {
        "precision": 0.0 if pd.isna(b.loc["High", "fraud_rate"]) else b.loc["High", "fraud_rate"],
        "recall": b.loc["High", "share_of_all_fraud"],
    }
    return iso, high


# =============================================================================
# a) ROC curve
# =============================================================================
def plot_roc_curve(scored, y, results):
    ranking = results["ranking"]
    iso, _ = operating_points(results)
    fig, ax = plt.subplots(figsize=(7.5, 6.5), layout="constrained")

    fpr, tpr, _ = roc_curve(y, scored["anomaly_score"])
    ax.plot(fpr, tpr, color=COLORS["if"], lw=2.5,
            label=f"Isolation Forest anomaly score (AUC = {ranking['if_roc_auc']:.3f})")

    # rule_score only takes 7 values (0-6), so its ROC curve has few steps
    fpr_r, tpr_r, _ = roc_curve(y, scored["rule_score"])
    ax.plot(fpr_r, tpr_r, color=COLORS["rules"], lw=2.2, ls="--", marker="o", ms=5,
            label=f"Rule score 0-6 (AUC = {ranking['rule_roc_auc']:.3f})")

    ax.plot([0, 1], [0, 1], color=COLORS["reference"], lw=1.3, ls=":",
            label="Random guessing (AUC = 0.500)")

    # Where the model actually operates with contamination=0.05
    ax.scatter(iso["fpr"], iso["recall"], s=110, color=COLORS["if"], edgecolor="black",
               zorder=5, label=f"IF flag threshold (contamination = {CONTAMINATION:.0%})")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("False positive rate (share of genuine claims flagged)")
    ax.set_ylabel("True positive rate (share of fraud caught)")
    ax.legend(loc="lower right", fontsize=9)
    add_titles(fig, ax, "ROC Curve: Isolation Forest vs Rule Score",
               f"Held-out test set: {len(y):,} claims, {int(y.sum())} fraudulent")
    return save_figure(fig, "roc_curve.png")


# =============================================================================
# b) Precision-recall curve
# =============================================================================
def plot_precision_recall_curve(scored, y, results):
    ranking = results["ranking"]
    base_rate = results["base_rate"]
    iso, high = operating_points(results)
    fig, ax = plt.subplots(figsize=(7.5, 6.5), layout="constrained")

    precision, recall, _ = precision_recall_curve(y, scored["anomaly_score"])
    # Step plot matches how average precision (PR-AUC) is computed
    ax.plot(recall, precision, drawstyle="steps-post", color=COLORS["if"], lw=2.5,
            label=f"Isolation Forest anomaly score (PR-AUC = {ranking['if_pr_auc']:.3f})")

    ax.axhline(base_rate, color=COLORS["reference"], lw=1.5, ls="--",
               label=f"Random guessing = base fraud rate ({base_rate:.1%})")

    ax.scatter(iso["recall"], iso["precision"], s=110, color=COLORS["if"], edgecolor="black",
               zorder=5, label=f"IF flag threshold (contamination = {CONTAMINATION:.0%})")
    ax.scatter(high["recall"], high["precision"], s=130, marker="D",
               color=COLORS["high_bucket"], edgecolor="black", zorder=6,
               label=f"'High' risk bucket (IF AND rule_score >= {RULE_FLAG_THRESHOLD})")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.set_xlabel("Recall (share of fraud caught)")
    ax.set_ylabel("Precision (share of flagged claims that are fraud)")
    ax.legend(loc="upper right", fontsize=9)
    add_titles(fig, ax, "Precision-Recall Curve: Isolation Forest",
               f"Held-out test set; only {base_rate:.1%} of claims are fraud, so PR is the harder test")
    return save_figure(fig, "precision_recall_curve.png")


# =============================================================================
# c) Confusion matrix
# =============================================================================
def plot_confusion_matrix(results):
    cm = results["confusion"]                       # rows = actual, cols = predicted
    iso, _ = operating_points(results)
    row_share = cm / cm.sum(axis=1, keepdims=True)
    row_names = ["genuine", "fraud"]
    annot = np.array([[f"{cm[i, j]:,}\n({row_share[i, j]:.1%} of actual {row_names[i]})"
                       for j in range(2)] for i in range(2)])

    fig, ax = plt.subplots(figsize=(7.5, 6.2), layout="constrained")
    # Colour by row share (not raw count) so the small fraud row is not washed out
    # by the huge genuine row; the annotations still show the actual counts.
    sns.heatmap(row_share, annot=annot, fmt="", cmap="Blues", vmin=0, vmax=1, cbar=True,
                cbar_kws={"label": "Share of actual class", "format": PercentFormatter(xmax=1)},
                linewidths=2, linecolor="white", square=True, annot_kws={"fontsize": 12},
                xticklabels=["Normal", "Anomaly (flagged)"],
                yticklabels=["Genuine", "Fraud"], ax=ax)
    ax.set_xlabel("Predicted by Isolation Forest", fontsize=11)
    ax.set_ylabel("Actual label", fontsize=11)
    ax.tick_params(axis="y", rotation=0)
    add_titles(fig, ax, f"Confusion Matrix: Isolation Forest (contamination = {CONTAMINATION:.0%})",
               f"Test set - precision {iso['precision']:.1%}, recall {iso['recall']:.1%}")
    return save_figure(fig, "confusion_matrix.png")


# =============================================================================
# d) Risk bucket analysis
# =============================================================================
def plot_risk_buckets(results):
    b = results["buckets"].reindex(RISK_LEVELS)
    levels = list(b.index)
    colors = [BUCKET_COLORS[level] for level in levels]
    total_claims = b["claims"].sum()
    base_rate = results["base_rate"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.8), layout="constrained")

    # Left: how many claims land in each bucket (investigation workload)
    bars = ax1.bar(levels, b["claims"], color=colors, edgecolor="black", linewidth=0.6)
    ax1.bar_label(bars, labels=[f"{c:,}\n({c / total_claims:.1%} of claims)" for c in b["claims"]],
                  padding=4, fontsize=10)
    ax1.set_ylim(0, b["claims"].max() * 1.22)
    ax1.set_title("Claim volume per bucket", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Combined risk level")
    ax1.set_ylabel("Number of test claims")

    # Right: how concentrated fraud is in each bucket
    rates = b["fraud_rate"].fillna(0)
    bars2 = ax2.bar(levels, rates, color=colors, edgecolor="black", linewidth=0.6)
    labels = [
        "n/a" if pd.isna(rate) else
        f"{rate:.1%}\n{int(fraud)} of {int(claims):,} claims\n{share:.0%} of all fraud"
        for rate, fraud, claims, share in zip(b["fraud_rate"], b["fraud"], b["claims"],
                                              b["share_of_all_fraud"])
    ]
    ax2.bar_label(bars2, labels=labels, padding=4, fontsize=9.5)
    ax2.axhline(base_rate, color=COLORS["reference"], ls="--", lw=1.5,
                label=f"Overall fraud rate ({base_rate:.1%})")
    ax2.set_ylim(0, max(rates.max(), base_rate) * 1.45)
    ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax2.set_title("Fraud rate per bucket (precision)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Combined risk level")
    ax2.set_ylabel("Share of claims that are fraud")
    ax2.legend(loc="upper right", fontsize=9)

    fig.suptitle("Combined Risk Buckets: Claim Volume vs Fraud Concentration",
                 fontsize=14, fontweight="bold")
    fig.supxlabel(f"High = Isolation Forest anomaly AND rule_score >= {RULE_FLAG_THRESHOLD}   |   "
                  "Medium = only one of the two   |   Low = neither   (held-out test set)",
                  fontsize=9.5, color=SUBTITLE_COLOR)
    return save_figure(fig, "risk_bucket_analysis.png")


# =============================================================================
# e) Anomaly score distribution
# =============================================================================
def plot_anomaly_score_distribution(scored, y, model):
    scores = scored["anomaly_score"].to_numpy()
    # predict() flags a claim when score_samples < offset_; anomaly_score = -score_samples,
    # so the equivalent cut-off on this axis is -offset_.
    threshold = -model.offset_
    bins = np.histogram_bin_edges(scores, bins=45)

    fig, ax = plt.subplots(figsize=(9, 5.8), layout="constrained")
    for name, mask, color in [("Genuine", y == 0, COLORS["genuine"]),
                              ("Fraud", y == 1, COLORS["fraud"])]:
        subset = scores[mask]
        # stat="density" per class, so the 5% fraud class is as visible as the 95% class
        sns.histplot(x=subset, bins=bins, stat="density", kde=True, color=color, alpha=0.35,
                     element="step", linewidth=1, line_kws={"linewidth": 2.2}, ax=ax,
                     label=f"{name} (n = {len(subset):,}, median = {np.median(subset):.3f})")

    ax.axvline(threshold, color="black", ls="--", lw=1.6,
               label=f"Flag threshold (contamination = {CONTAMINATION:.0%})")
    x_max = ax.get_xlim()[1]
    ax.axvspan(threshold, x_max, color="#BBBBBB", alpha=0.18, zorder=0)
    ax.set_xlim(right=x_max)
    ax.text(threshold, ax.get_ylim()[1] * 0.97, "  flagged as anomaly ->", va="top",
            ha="left", fontsize=9.5, color=SUBTITLE_COLOR)

    ax.set_xlabel("Anomaly score (higher = more unusual)")
    ax.set_ylabel("Density (each class scaled separately)")
    ax.legend(loc="upper left", fontsize=9)
    add_titles(fig, ax, "Anomaly Score Distribution: Fraud vs Genuine Claims",
               "Held-out test set - the more the curves overlap, the harder fraud is to separate")
    return save_figure(fig, "anomaly_score_distribution.png")


# =============================================================================
# f) Feature correlation with is_fraud
# =============================================================================
def compute_feature_correlations(data):
    """Pearson correlation of engineered features with is_fraud (analysis only).

    Preferred: reuse feature_engineering.correlation_report on the UNSCALED engineered
    features, which gives exactly the numbers its summary printed.
    Fallback: recompute from the processed features file. Binary flags and linearly
    scaled columns give the same correlation; log-transformed columns
    (claim_to_avg_ratio) can differ slightly.
    """
    try:
        import feature_engineering as fe
        claims, _ = fe.load_data()
        _, _, engineered = fe.preprocess(claims)          # recomputes in memory, writes nothing
        labels = (pd.read_csv(LABELS_CSV).set_index(ID_COL)
                  .loc[engineered[ID_COL]].reset_index())  # align rows by claim_id
        corr = fe.correlation_report(engineered, labels)["pearson"]
        source = "feature_engineering.correlation_report"
    except Exception as exc:                               # noqa: BLE001 - fall back on anything
        print(f"[feature_correlation] Could not reuse feature_engineering ({exc!r}); "
              "recomputing from processed files.")
        labels = pd.read_csv(LABELS_CSV)
        df = data
        if "provider_fraud_rate" in labels.columns:
            df = data.merge(labels[[ID_COL, "provider_fraud_rate"]], on=ID_COL, how="left")
            df = df.rename(columns={"provider_fraud_rate": "provider_fraud_rate (out-of-fold)"})
        cols = [c for c in ENGINEERED_FEATURES + ["provider_fraud_rate (out-of-fold)"]
                if c in df.columns]
        corr = df[cols].corrwith(df[LABEL_COL])
        source = "processed features + labels files"

    corr = corr[~corr.index.str.contains("LEAKY")].dropna()   # never chart the leaky version
    top = corr.reindex(corr.abs().sort_values(ascending=False).index).head(TOP_N_FEATURES)
    return top, source


def plot_feature_correlation(data):
    top, source = compute_feature_correlations(data)
    plot_values = top.iloc[::-1]                    # barh draws bottom-up: strongest on top
    colors = [COLORS["positive"] if v > 0 else COLORS["negative"] for v in plot_values]

    fig, ax = plt.subplots(figsize=(9.5, 6.2), layout="constrained")
    bars = ax.barh(plot_values.index, plot_values.values, color=colors,
                   edgecolor="black", linewidth=0.5)
    ax.bar_label(bars, labels=[f"{v:+.3f}" for v in plot_values.values], padding=4, fontsize=9.5)
    ax.axvline(0, color="black", lw=0.9)

    limit = max(plot_values.abs().max() * 1.25, 0.05)
    ax.set_xlim(-limit if (plot_values < 0).any() else -limit * 0.1, limit)
    ax.set_xlabel("Pearson correlation with is_fraud")
    ax.set_ylabel("")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Patch(color=COLORS["positive"], label="Positive: higher value -> more fraud"),
                       Patch(color=COLORS["negative"], label="Negative: higher value -> less fraud")],
              loc="lower right", fontsize=9)
    add_titles(fig, ax, f"Top {len(top)} Engineered Features by Correlation with Fraud",
               f"All {len(data):,} claims - analysis only, not used for training "
               f"(source: {source})")
    return save_figure(fig, "feature_correlation.png")


# =============================================================================
# Main
# =============================================================================
def main():
    set_style()
    data, model, scored, y_test, results = reproduce_results()

    print(f"Test set: {results['n_test']:,} claims, {results['n_fraud']} fraud | "
          f"IF ROC-AUC {results['ranking']['if_roc_auc']:.3f} | "
          f"PR-AUC {results['ranking']['if_pr_auc']:.3f}  (should match train_model.py)")

    saved = [
        plot_roc_curve(scored, y_test, results),
        plot_precision_recall_curve(scored, y_test, results),
        plot_confusion_matrix(results),
        plot_risk_buckets(results),
        plot_anomaly_score_distribution(scored, y_test, model),
        plot_feature_correlation(data),
    ]

    print(f"\nSaved {len(saved)} figures:")
    for path in saved:
        print(f"  {path}")


if __name__ == "__main__":
    main()