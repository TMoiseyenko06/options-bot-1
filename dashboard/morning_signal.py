"""
dashboard/morning_signal.py — Morning signal dashboard.

THIS IS THE ONLY FILE THAT USES DATABENTO_API_KEY AND MASSIVE_API_KEY.
All other modules read from local files only.

Fetches the last 30 days of SPY data via Databento API,
fetches VIX/sectors via yfinance, fetches real options chain pricing
via Massive API, generates today's signal with real strike/debit details.
"""

import json
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import xgboost as xgb
import shap
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env from project root (never commit .env to git)
load_dotenv(PROJECT_ROOT / ".env")

from dashboard.telegram import send_signal, send_error

MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

from features.engineer import (
    engineer_features,
    FEATURE_COLUMNS,
    FOMC_DATES,
    CPI_DATES,
    _days_until_next,
    _find_col,
)

DATABENTO_API_KEY = os.environ.get("DATABENTO_API_KEY")
MASSIVE_API_KEY = os.environ.get("MASSIVE_API_KEY")
MASSIVE_BASE_URL = "https://api.massive.com"

LABEL_INV = {0: -1, 1: 0, 2: 1}
SIGNAL_LABELS = {1: "BUY CALL SPREAD", -1: "BUY PUT SPREAD", 0: "NO TRADE"}

# How many calendar days back to fetch from live API
LOOKBACK_DAYS = 30

# FOMC / CPI proximity filters
FOMC_PROXIMITY_DAYS = 1    # skip if FOMC is tomorrow or today
CPI_PROXIMITY_DAYS = 0     # skip on CPI day itself


# ── Databento live fetch ──────────────────────────────────────────────────────

def _fetch_databento_live(symbol: str, dataset: str, schema: str, lookback: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """
    Fetch the last `lookback` calendar days of data for a symbol via Databento API.
    Only called from morning_signal.py.
    """
    if DATABENTO_API_KEY is None:
        raise EnvironmentError(
            "[morning_signal] DATABENTO_API_KEY environment variable is not set."
        )

    import databento as db  # import here to make dependency explicit

    client = db.Historical(DATABENTO_API_KEY)
    end_dt = datetime.utcnow().date()
    start_dt = end_dt - timedelta(days=lookback)

    print(f"  [API] Fetching {symbol} ({dataset}/{schema}) {start_dt} → {end_dt} …")
    data = client.timeseries.get_range(
        dataset=dataset,
        symbols=[symbol],
        schema=schema,
        start=start_dt.isoformat(),
        end=end_dt.isoformat(),
    )
    df = data.to_df()
    if df.empty:
        print(f"  [API] WARNING: No data returned for {symbol}")
        return pd.DataFrame()

    # Normalise
    ts_col = next((c for c in ("ts_event", "ts_recv") if c in df.columns), None)
    if ts_col:
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
        df["date"] = df[ts_col].dt.normalize().dt.tz_localize(None)
    elif isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        df.rename(columns={df.columns[0]: "ts_event"}, inplace=True)
        df["date"] = pd.to_datetime(df["ts_event"]).dt.normalize()

    # Aggregate to daily OHLCV
    agg = {}
    for col, func in [("open", "first"), ("high", "max"), ("low", "min"), ("close", "last"), ("volume", "sum")]:
        if col in df.columns:
            agg[col] = func
    if agg:
        df = df.groupby("date").agg(agg).reset_index()

    print(f"  [API] Got {len(df)} rows")
    return df


def _fetch_es_live() -> pd.DataFrame:
    """Fetch ES continuous front-month futures (GLBX.MDP3 / ohlcv-1d)."""
    return _fetch_databento_live(
        symbol="ES.c.0",
        dataset="GLBX.MDP3",
        schema="ohlcv-1d",
    )


def _fetch_spy_live() -> pd.DataFrame:
    """Fetch SPY daily OHLCV via Databento (XNAS.ITCH / ohlcv-1d)."""
    return _fetch_databento_live(
        symbol="SPY",
        dataset="XNAS.ITCH",
        schema="ohlcv-1d",
    )


# ── yfinance fetches ──────────────────────────────────────────────────────────

def _fetch_yf_last30(tickers: list[str]) -> pd.DataFrame:
    end = datetime.today()
    start = end - timedelta(days=LOOKBACK_DAYS + 10)   # small buffer for weekends

    raw = yf.download(
        tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )

    if raw.empty:
        return pd.DataFrame()

    # Flatten multi-level columns
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = ["_".join(c).lower() for c in raw.columns]
    else:
        raw.columns = [c.lower() for c in raw.columns]

    raw = raw.reset_index()
    raw = raw.rename(columns={"Date": "date", "index": "date"})
    raw["date"] = pd.to_datetime(raw["date"]).dt.normalize()
    return raw


# ── Feature computation for today ────────────────────────────────────────────

def build_today_features(
    spy_df: pd.DataFrame,
    es_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    sectors_df: pd.DataFrame,
    existing_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Append the latest rows to existing_features and recompute rolling features.
    Returns the last row (today) as a single-row DataFrame.
    """
    # ── Merge new data into a daily frame ────────────────────────────────────
    spy_df = spy_df.copy()
    es_df = es_df.copy()
    vix_df = vix_df.copy()
    sectors_df = sectors_df.copy()

    # Rename columns to match existing schema
    spy_df = spy_df.rename(columns={c: f"spy_{c}" for c in spy_df.columns if c != "date"})
    es_df = es_df.rename(columns={c: f"es_{c}" for c in es_df.columns if c != "date"})

    # Align VIX columns
    for col in list(vix_df.columns):
        if col != "date":
            ticker = col.split("_")[0] if "_" in col else "vix"
            suffix = col.split("_", 1)[1] if "_" in col else col
            # Map ^VIX → vix_, ^VIX9D → vix9d_
            if "vix9d" in col.lower() or "^vix9d" in col.lower():
                vix_df = vix_df.rename(columns={col: f"vix9d_{suffix}"})
            elif "vix" in col.lower():
                vix_df = vix_df.rename(columns={col: f"vix_{suffix}"})

    # Align sector columns
    for ticker in ["xlk", "xlf", "xle", "xlv"]:
        for col in list(sectors_df.columns):
            if col != "date" and ticker in col.lower() and not col.startswith(f"{ticker}_"):
                new_col = col.lower().replace(f"^{ticker}", ticker).replace(ticker, f"{ticker}", 1)
                if not new_col.startswith(f"{ticker}_"):
                    new_col = f"{ticker}_{col.split('_', 1)[-1]}" if "_" in col else f"{ticker}_{col}"
                sectors_df = sectors_df.rename(columns={col: new_col})

    # Merge all onto date
    new_rows = spy_df.merge(es_df, on="date", how="left")
    new_rows = new_rows.merge(vix_df, on="date", how="left")
    new_rows = new_rows.merge(sectors_df, on="date", how="left")

    # Combine with existing features (keep raw columns only)
    raw_cols = [c for c in existing_features.columns if c not in FEATURE_COLUMNS
                and c not in ("target", "target_idx", "spy_intraday_return")]
    existing_raw = existing_features[["date"] + [c for c in raw_cols if c in existing_features.columns]]

    combined = pd.concat([existing_raw, new_rows], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

    # Compute overnight gap for ES if available
    if "es_close" in combined.columns and "es_open" in combined.columns:
        combined["es_overnight_gap"] = (
            (combined["es_open"] - combined["es_close"].shift(1)) / combined["es_close"].shift(1)
        )

    # Re-engineer features
    combined = engineer_features(combined)

    # Return only the last row (today)
    return combined.tail(1).copy()


# ── Signal generation ─────────────────────────────────────────────────────────

def generate_signal(today_row: pd.DataFrame, model: xgb.XGBClassifier, feature_cols: list[str]) -> dict:
    """Run prediction and compute SHAP for today's features."""
    available = [c for c in feature_cols if c in today_row.columns]
    X = today_row[available].values.astype(np.float32)

    proba = model.predict_proba(X)[0]   # (3,) probabilities
    pred_class = int(np.argmax(proba))
    pred_direction = LABEL_INV[pred_class]
    confidence = float(proba[pred_class])

    # SHAP for today
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # SHAP for predicted class
    if isinstance(shap_values, list):
        sv = shap_values[pred_class][0]
    elif shap_values.ndim == 3:
        sv = shap_values[0, :, pred_class]
    else:
        sv = shap_values[0]

    # Top 5 features by |SHAP|
    shap_df = pd.DataFrame({
        "feature": available,
        "shap_value": sv,
        "feature_value": X[0],
    }).sort_values("shap_value", key=abs, ascending=False).head(5)

    # Filters
    today_date = pd.to_datetime(today_row["date"].values[0])
    days_fomc = _days_until_next(FOMC_DATES, pd.Series([today_date]))[0]
    days_cpi = _days_until_next(CPI_DATES, pd.Series([today_date]))[0]

    active_filters = []
    signal_override = False

    if days_fomc <= FOMC_PROXIMITY_DAYS:
        active_filters.append(f"FOMC in {days_fomc} day(s) — caution")
        signal_override = True

    if days_cpi <= CPI_PROXIMITY_DAYS:
        active_filters.append(f"CPI release today — signal suppressed")
        signal_override = True

    final_direction = 0 if signal_override else pred_direction

    vix_col = _find_col(today_row, ["vix_level", "vix_close", "vix_vix_close"], required=False)
    vix9d_col = _find_col(today_row, ["vix9d_minus_vix", "vix9d_close"], required=False)

    vix_val = float(today_row[vix_col].values[0]) if vix_col else None
    vix9d_spread = float(today_row["vix9d_minus_vix"].values[0]) if "vix9d_minus_vix" in today_row.columns else None

    return {
        "date": str(today_date.date()),
        "signal": SIGNAL_LABELS[final_direction],
        "raw_signal": SIGNAL_LABELS[pred_direction],
        "confidence": confidence,
        "probabilities": {"DOWN": float(proba[0]), "FLAT": float(proba[1]), "UP": float(proba[2])},
        "top_shap_features": shap_df.to_dict(orient="records"),
        "vix": vix_val,
        "vix9d_minus_vix": vix9d_spread,
        "days_to_fomc": days_fomc,
        "days_to_cpi": days_cpi,
        "active_filters": active_filters,
        "signal_overridden": signal_override,
    }


# ── Terminal output ───────────────────────────────────────────────────────────

def print_signal(result: dict):
    line = "=" * 60
    print(f"\n{line}")
    print(f"  0DTE SPX DEBIT SPREAD — MORNING SIGNAL")
    print(f"  Date: {result['date']}")
    print(line)

    # Signal
    sig = result["signal"]
    if sig == "NO TRADE":
        color_code = "\033[93m"   # yellow
    elif "CALL" in sig:
        color_code = "\033[92m"   # green
    else:
        color_code = "\033[91m"   # red
    reset = "\033[0m"

    print(f"\n  SIGNAL: {color_code}{sig}{reset}")
    print(f"  Confidence: {result['confidence']:.1%}")

    if result["signal_overridden"]:
        print(f"  (Raw model signal was: {result['raw_signal']} — overridden by filter)")

    print(f"\n  Probabilities:")
    for label, prob in result["probabilities"].items():
        bar = "█" * int(prob * 20)
        print(f"    {label:5s}: {prob:.1%} {bar}")

    print(f"\n  Top 5 SHAP features driving today's signal:")
    print(f"  {'Feature':<30} {'Value':>10} {'SHAP':>10}")
    print(f"  {'-'*52}")
    for row in result["top_shap_features"]:
        direction_arrow = "▲" if row["shap_value"] > 0 else "▼"
        print(f"  {row['feature']:<30} {row['feature_value']:>10.4f} {row['shap_value']:>+9.4f} {direction_arrow}")

    print(f"\n  Market Context:")
    if result["vix"] is not None:
        print(f"    VIX level      : {result['vix']:.2f}")
    if result["vix9d_minus_vix"] is not None:
        spread_label = "term structure inverted" if result["vix9d_minus_vix"] < 0 else "normal"
        print(f"    VIX9D-VIX      : {result['vix9d_minus_vix']:.2f}  ({spread_label})")
    print(f"    Days to FOMC   : {result['days_to_fomc']}")
    print(f"    Days to CPI    : {result['days_to_cpi']}")

    if result["active_filters"]:
        print(f"\n  Active filters:")
        for f in result["active_filters"]:
            print(f"    ⚠  {f}")

    print(f"\n{line}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("[morning_signal] Starting morning signal dashboard …")

    # Verify API key
    if not DATABENTO_API_KEY:
        print("[morning_signal] ERROR: DATABENTO_API_KEY environment variable not set.")
        sys.exit(1)

    # Load model
    model_path = MODELS_DIR / "best_model.json"
    feature_list_path = MODELS_DIR / "feature_list.json"

    if not model_path.exists():
        print(f"[morning_signal] ERROR: Model not found at {model_path}")
        print("  Run: python models/train.py")
        sys.exit(1)

    print("[morning_signal] Loading model …")
    model = xgb.XGBClassifier()
    model.load_model(str(model_path))

    with open(feature_list_path) as f:
        meta = json.load(f)
    feature_cols = meta["features"]

    # Load existing features for rolling window context
    features_path = PROCESSED_DIR / "features.parquet"
    if not features_path.exists():
        print(f"[morning_signal] ERROR: {features_path} not found.")
        print("  Run the full pipeline first (parse_dbn → fetch_yfinance → merge → engineer).")
        sys.exit(1)

    print("[morning_signal] Loading existing feature history …")
    existing_features = pd.read_parquet(features_path)
    existing_features["date"] = pd.to_datetime(existing_features["date"])

    # Fetch live data from Databento API (ONLY place API is used)
    print("\n[morning_signal] Fetching live data via Databento API …")
    try:
        es_live = _fetch_es_live()
    except Exception as e:
        print(f"  WARNING: ES futures fetch failed: {e}")
        es_live = pd.DataFrame()

    try:
        spy_live = _fetch_spy_live()
    except Exception as e:
        print(f"  WARNING: SPY fetch failed: {e}")
        spy_live = pd.DataFrame()

    # Fetch VIX and sectors via yfinance
    print("\n[morning_signal] Fetching VIX and sectors via yfinance …")
    vix_sectors_df = _fetch_yf_last30(["^VIX", "^VIX9D", "XLK", "XLF", "XLE", "XLV"])

    # Split into VIX and sectors
    vix_cols = ["date"] + [c for c in vix_sectors_df.columns if any(
        t in c.lower() for t in ["vix"]
    )]
    sector_cols = ["date"] + [c for c in vix_sectors_df.columns if any(
        t in c.lower() for t in ["xlk", "xlf", "xle", "xlv"]
    )]

    vix_live = vix_sectors_df[vix_cols] if vix_cols else pd.DataFrame()
    sectors_live = vix_sectors_df[sector_cols] if sector_cols else pd.DataFrame()

    # Use fallback SPY from existing if live fetch failed
    if spy_live.empty:
        print("  WARNING: Using existing SPY data as fallback.")
        spy_live = existing_features[["date", "spy_open", "spy_close", "spy_high", "spy_low"]].tail(30)
        spy_live = spy_live.rename(columns={c: c.replace("spy_", "") for c in spy_live.columns if c != "date"})

    if es_live.empty:
        print("  WARNING: Using existing ES data as fallback.")
        es_cols = ["date"] + [c for c in existing_features.columns if c.startswith("es_")]
        if es_cols:
            es_live = existing_features[es_cols].tail(30)
            es_live = es_live.rename(columns={c: c.replace("es_", "") for c in es_live.columns if c != "date"})

    # Build today's features
    print("\n[morning_signal] Computing today's features …")
    try:
        today_features = build_today_features(
            spy_live, es_live, vix_live, sectors_live, existing_features
        )
    except Exception as e:
        print(f"[morning_signal] ERROR building features: {e}")
        import traceback
        traceback.print_exc()
        send_error(f"Feature build failed:\n{e}")
        sys.exit(1)

    # Generate signal
    print("[morning_signal] Generating signal …")
    result = generate_signal(today_features, model, feature_cols)

    # Print to terminal
    print_signal(result)

    # Send to Telegram
    send_signal(result)

    # Save today's signal
    sig_path = PROJECT_ROOT / "dashboard" / "last_signal.json"
    with open(sig_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"[morning_signal] Signal saved → {sig_path}")


if __name__ == "__main__":
    main()
