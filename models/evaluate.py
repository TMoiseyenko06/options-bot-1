"""
models/evaluate.py — Evaluate trained model: classification report, confusion matrix,
SHAP summary plot, and SHAP single-day explanation plot.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
import shap
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

TEST_START = "2023-01-01"

# Label display names (0-indexed XGBoost → human)
CLASS_NAMES = {0: "DOWN (-1)", 1: "FLAT (0)", 2: "UP (+1)"}
LABEL_INV = {0: -1, 1: 0, 2: 1}


def load_model_and_features():
    model_path = MODELS_DIR / "best_model.json"
    feature_list_path = MODELS_DIR / "feature_list.json"

    if not model_path.exists():
        raise FileNotFoundError(f"[evaluate] Model not found: {model_path}\nRun train.py first.")
    if not feature_list_path.exists():
        raise FileNotFoundError(f"[evaluate] Feature list not found: {feature_list_path}\nRun train.py first.")

    model = xgb.XGBClassifier()
    model.load_model(str(model_path))

    with open(feature_list_path) as f:
        meta = json.load(f)

    return model, meta["features"], meta.get("label_map", {})


def load_test_data(feature_cols: list[str]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    path = PROCESSED_DIR / "features.parquet"
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= TEST_START].copy()

    label_map = {-1: 0, 0: 1, 1: 2}
    df["target_idx"] = df["target"].map(label_map)
    df = df.dropna(subset=feature_cols + ["target_idx"])

    X = df[feature_cols].values.astype(np.float32)
    y = df["target_idx"].values.astype(int)
    return df, X, y


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_dir: Path):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    labels = [CLASS_NAMES[i] for i in [0, 1, 2]]

    # Normalised version
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    text = [[f"{cm[i][j]}<br>{cm_norm[i][j]:.1%}" for j in range(3)] for i in range(3)]

    fig = ff.create_annotated_heatmap(
        z=cm_norm.tolist(),
        x=labels,
        y=labels,
        annotation_text=text,
        colorscale="Blues",
        showscale=True,
    )
    fig.update_layout(
        title="Confusion Matrix (Test Set 2023–present)",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        yaxis_autorange="reversed",
        width=600,
        height=500,
    )

    out = save_dir / "confusion_matrix.html"
    fig.write_html(str(out))
    print(f"[evaluate] Confusion matrix saved → {out}")
    return fig


def plot_shap_summary(model: xgb.XGBClassifier, X: np.ndarray, feature_names: list[str], save_dir: Path):
    """SHAP summary plot – top 10 features, mean |SHAP| across all classes."""
    print("[evaluate] Computing SHAP values …")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # shap_values shape: (n_samples, n_features, n_classes) or list of arrays
    if isinstance(shap_values, list):
        # list of (n_samples, n_features) per class
        mean_abs = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    else:
        mean_abs = np.mean(np.abs(shap_values), axis=0) if shap_values.ndim == 3 else np.abs(shap_values)

    mean_abs_per_feature = mean_abs.mean(axis=0) if mean_abs.ndim == 2 else mean_abs

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_per_feature,
    }).sort_values("mean_abs_shap", ascending=False).head(10)

    fig = go.Figure(go.Bar(
        x=importance_df["mean_abs_shap"],
        y=importance_df["feature"],
        orientation="h",
        marker_color="steelblue",
    ))
    fig.update_layout(
        title="SHAP Feature Importance — Top 10 (Mean |SHAP| across all classes)",
        xaxis_title="Mean |SHAP value|",
        yaxis_title="Feature",
        yaxis_autorange="reversed",
        height=500,
        width=800,
    )

    out = save_dir / "shap_summary.html"
    fig.write_html(str(out))
    print(f"[evaluate] SHAP summary plot saved → {out}")
    return fig, explainer, shap_values


def plot_shap_single_day(
    explainer,
    shap_values,
    X: np.ndarray,
    df: pd.DataFrame,
    feature_names: list[str],
    save_dir: Path,
    row_idx: int = -1,
):
    """
    SHAP waterfall-style explanation for a single day.
    Default: last row in the test set.
    """
    if row_idx == -1:
        row_idx = len(X) - 1

    date_label = str(df.iloc[row_idx]["date"].date()) if "date" in df.columns else f"row {row_idx}"
    pred_class = int(np.argmax(
        explainer.model.inplace_predict(X[row_idx:row_idx+1])[0]
    ))
    pred_label = CLASS_NAMES.get(pred_class, str(pred_class))

    # Get SHAP values for the predicted class
    if isinstance(shap_values, list):
        sv_day = shap_values[pred_class][row_idx]
    elif shap_values.ndim == 3:
        sv_day = shap_values[row_idx, :, pred_class]
    else:
        sv_day = shap_values[row_idx]

    feat_df = pd.DataFrame({
        "feature": feature_names,
        "shap_value": sv_day,
        "feature_value": X[row_idx],
    }).sort_values("shap_value", key=abs, ascending=True)

    colors = ["crimson" if v < 0 else "steelblue" for v in feat_df["shap_value"]]

    fig = go.Figure(go.Bar(
        x=feat_df["shap_value"],
        y=feat_df["feature"],
        orientation="h",
        marker_color=colors,
        text=[f"{fv:.4f}" for fv in feat_df["feature_value"]],
        textposition="outside",
    ))
    fig.update_layout(
        title=f"SHAP Explanation — {date_label} | Prediction: {pred_label}",
        xaxis_title="SHAP value (impact on model output)",
        yaxis_title="Feature",
        height=600,
        width=900,
    )

    out = save_dir / "shap_single_day.html"
    fig.write_html(str(out))
    print(f"[evaluate] SHAP single-day plot saved → {out}")
    return fig


def main():
    print("[evaluate] Loading model and features …")
    model, feature_cols, label_map = load_model_and_features()

    print("[evaluate] Loading test data …")
    df, X, y = load_test_data(feature_cols)
    print(f"  Test rows: {len(df):,}  ({df['date'].min().date()} → {df['date'].max().date()})")

    # ── Classification report ─────────────────────────────────────────────────
    y_pred = model.predict(X)
    target_names = [CLASS_NAMES[i] for i in [0, 1, 2]]
    report = classification_report(y, y_pred, target_names=target_names, zero_division=0)
    print("\n[evaluate] Classification Report (Test Set):")
    print(report)

    # Save report
    report_path = MODELS_DIR / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Test Set: {TEST_START} -> present\n\n")
        f.write(report)
    print(f"[evaluate] Report saved → {report_path}")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    plot_confusion_matrix(y, y_pred, MODELS_DIR)

    # ── SHAP ─────────────────────────────────────────────────────────────────
    fig_summary, explainer, shap_values = plot_shap_summary(model, X, feature_cols, MODELS_DIR)
    plot_shap_single_day(explainer, shap_values, X, df, feature_cols, MODELS_DIR)

    print("\n[evaluate] Done.")


if __name__ == "__main__":
    main()
