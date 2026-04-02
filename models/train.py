"""
models/train.py — Train XGBoost classifier on engineered features.

Uses CUDA GPU, TimeSeriesSplit CV, and a grid search over key hyperparameters.
Saves best model to models/best_model.json and feature list to models/feature_list.json.
"""

import json
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import precision_score
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_END = "2022-12-31"
TEST_START = "2023-01-01"
N_CV_FOLDS = 5

# Label mapping: XGBoost requires 0-indexed classes
LABEL_MAP = {-1: 0, 0: 1, 1: 2}
LABEL_INV = {v: k for k, v in LABEL_MAP.items()}

FEATURE_COLUMNS = [
    "spy_prev_return",
    "spy_5d_return",
    "spy_dist_ma20",
    "es_overnight_gap",
    "vix_level",
    "vix_vs_ma10",
    "vix9d_minus_vix",
    "iv_rv_spread",
    "xlk_vs_spy_1d",
    "xlk_vs_spy_5d",
    "xlf_vs_spy_1d",
    "sectors_positive_count",
    "day_of_week",
    "week_of_month",
    "days_to_fomc",
    "days_to_cpi",
]

# Grid search parameter space
PARAM_GRID = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [200, 500, 1000],
    "subsample": [0.7, 0.9],
    "colsample_bytree": [0.7, 0.9],
}

BASE_PARAMS = {
    "device": "cuda",
    "tree_method": "hist",
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "early_stopping_rounds": 50,
    "verbosity": 0,
}


def _try_cuda_fallback_cpu(params: dict) -> dict:
    """Return params with device=cpu if CUDA is not available."""
    try:
        # Strip early_stopping_rounds for this probe — it requires an eval_set
        probe_params = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
        test = xgb.XGBClassifier(**{**probe_params, "n_estimators": 10})
        X_probe = np.random.rand(20, 4).astype(np.float32)
        y_probe = np.random.randint(0, 3, 20)
        test.fit(X_probe, y_probe)
        return params
    except Exception as e:
        if "cuda" in str(e).lower() or "gpu" in str(e).lower():
            print("[train] CUDA not available, falling back to CPU.")
            p = params.copy()
            p["device"] = "cpu"
            return p
        raise


def _precision_non_neutral(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Precision score averaged over classes 1 (DOWN) and 2 (UP) only,
    ignoring class 1 (NEUTRAL) in our 0-indexed scheme.
    LABEL_MAP: -1→0, 0→1, 1→2
    """
    # Classes 0 (was -1) and 2 (was +1) are the directional classes
    prec = precision_score(y_true, y_pred, labels=[0, 2], average="macro", zero_division=0)
    return prec


def load_data():
    path = PROCESSED_DIR / "features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"[train] {path} not found. Run engineer.py first.")

    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])

    # Select only available feature columns
    available_features = [c for c in FEATURE_COLUMNS if c in df.columns]
    missing = set(FEATURE_COLUMNS) - set(available_features)
    if missing:
        print(f"[train] WARNING: Missing features (will use NaN): {missing}")

    df = df.dropna(subset=available_features + ["target"])

    # Map labels to 0-indexed
    df["target_idx"] = df["target"].map(LABEL_MAP)

    return df, available_features


def run_cv_for_params(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: dict,
    n_folds: int = N_CV_FOLDS,
) -> float:
    """Run TimeSeriesSplit CV and return mean directional precision."""
    tscv = TimeSeriesSplit(n_splits=n_folds)
    precisions = []

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        y_pred = model.predict(X_val)
        prec = _precision_non_neutral(y_val, y_pred)
        precisions.append(prec)

    return float(np.mean(precisions))


def grid_search(X_train: np.ndarray, y_train: np.ndarray, base_params: dict):
    """Grid search over PARAM_GRID; return best params and score."""
    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    combos = list(product(*values))

    print(f"[train] Grid search: {len(combos)} combinations × {N_CV_FOLDS} folds")
    best_score = -np.inf
    best_params = None

    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        full_params = {**base_params, **params}

        try:
            score = run_cv_for_params(X_train, y_train, full_params)
        except Exception as e:
            print(f"  Combo {i+1}/{len(combos)} ERROR: {e}")
            continue

        if score > best_score:
            best_score = score
            best_params = params
            print(f"  ✓ New best  combo {i+1}/{len(combos)}  precision={score:.4f}  {params}")
        elif (i + 1) % 20 == 0:
            print(f"  … combo {i+1}/{len(combos)}  current best={best_score:.4f}")

    return best_params, best_score


def train_final_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    best_params: dict,
    base_params: dict,
):
    """Train final model on full train set with best params."""
    final_params = {**base_params, **best_params}
    model = xgb.XGBClassifier(**final_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=100,
    )
    return model


def main():
    print("[train] Loading data …")
    df, feature_cols = load_data()

    train_df = df[df["date"] <= TRAIN_END].copy()
    test_df = df[df["date"] >= TEST_START].copy()

    print(f"[train] Train rows: {len(train_df):,}  ({train_df['date'].min().date()} → {train_df['date'].max().date()})")
    print(f"[train] Test  rows: {len(test_df):,}  ({test_df['date'].min().date()} → {test_df['date'].max().date()})")

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df["target_idx"].values.astype(int)
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df["target_idx"].values.astype(int)

    print(f"\n[train] Train label distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"[train] Test  label distribution: {dict(zip(*np.unique(y_test, return_counts=True)))}")

    # Try CUDA, fall back to CPU if unavailable
    base_params = _try_cuda_fallback_cpu(BASE_PARAMS.copy())

    print("\n[train] Starting grid search …")
    best_params, best_cv_score = grid_search(X_train, y_train, base_params)

    print(f"\n[train] Best CV precision (directional): {best_cv_score:.4f}")
    print(f"[train] Best params: {best_params}")

    print("\n[train] Training final model on full train set …")
    model = train_final_model(X_train, y_train, X_test, y_test, best_params, base_params)

    # ── Save model ────────────────────────────────────────────────────────────
    model_path = MODELS_DIR / "best_model.json"
    model.save_model(str(model_path))
    print(f"\n[train] ✓ Model saved → {model_path}")

    # ── Save feature list ─────────────────────────────────────────────────────
    feature_list_path = MODELS_DIR / "feature_list.json"
    with open(feature_list_path, "w") as f:
        json.dump({"features": feature_cols, "label_map": LABEL_MAP}, f, indent=2)
    print(f"[train] ✓ Feature list saved → {feature_list_path}")

    # ── Quick test-set evaluation ─────────────────────────────────────────────
    y_pred = model.predict(X_test)
    test_precision = _precision_non_neutral(y_test, y_pred)
    print(f"\n[train] Test-set directional precision: {test_precision:.4f}")

    return model, feature_cols


if __name__ == "__main__":
    main()
