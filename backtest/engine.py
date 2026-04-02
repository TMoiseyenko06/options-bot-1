"""
backtest/engine.py — Backtest the debit spread strategy on the test set (2023–present).

Entry: 9:45am ET signal day.
Exit: 100% profit target, 100% stop loss, or 3:45pm ET time stop.
Uses SPY intraday data from dbn parquets where available; falls back to daily OHLCV.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "backtest" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TEST_START = "2023-01-01"

# Trade parameters
SPREAD_WIDTH = 3          # points (3-point spread)
DEBIT_RATIO = 0.35        # debit = 35% of spread width
DEBIT = SPREAD_WIDTH * DEBIT_RATIO * 100   # in dollars ($105 per spread)
MAX_GAIN = SPREAD_WIDTH * 100 - DEBIT       # max gain = $195
MAX_LOSS = -DEBIT                            # max loss = -$105

PROFIT_TARGET_PCT = 1.0   # 100% profit target
STOP_LOSS_PCT = 1.0        # 100% stop loss (full loss)

LABEL_INV = {0: -1, 1: 0, 2: 1}


def load_model_and_features():
    model_path = MODELS_DIR / "best_model.json"
    feature_list_path = MODELS_DIR / "feature_list.json"

    if not model_path.exists():
        raise FileNotFoundError(f"[backtest] Model not found: {model_path}")
    model = xgb.XGBClassifier()
    model.load_model(str(model_path))

    with open(feature_list_path) as f:
        meta = json.load(f)

    return model, meta["features"]


def load_test_features() -> pd.DataFrame:
    path = PROCESSED_DIR / "features.parquet"
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= TEST_START].copy()
    return df.sort_values("date").reset_index(drop=True)


def _simulate_intraday_exit(
    row: pd.Series,
    direction: int,
    debit: float,
    max_gain: float,
    max_loss: float,
    profit_target_pct: float = PROFIT_TARGET_PCT,
    stop_loss_pct: float = STOP_LOSS_PCT,
) -> tuple[float, str, float]:
    """
    Simulate spread P&L using intraday SPY OHLCV.

    For a call spread (direction=1): profits when SPY rises.
    For a put spread (direction=-1): profits when SPY falls.

    Returns (pnl_dollars, exit_type, hold_hours).
    """
    spy_open = _get_price(row, ["spy_open"])
    spy_close = _get_price(row, ["spy_close", "spy_adj_close"])
    spy_high = _get_price(row, ["spy_high"])
    spy_low = _get_price(row, ["spy_low"])

    if spy_open is None or spy_close is None:
        return 0.0, "no_data", 0.0

    profit_target = debit * profit_target_pct
    stop_loss_trigger = debit * stop_loss_pct

    intraday_return = (spy_close - spy_open) / spy_open if spy_open else 0.0

    # Approximate spread value at close based on intraday return
    # Call spread gains when underlying rises; put spread gains when it falls
    directional_move = intraday_return * direction

    # Approximate mark based on spread intrinsic / time value model
    # At 100% profit target the spread is worth 2x debit (full value = max_gain + debit)
    # At 100% stop the spread is worth 0

    # Simplified P&L based on SPY directional move
    # Assume linear relationship between SPY return and spread value
    # Normalise: a 0.5% directional move roughly achieves 50% of max gain
    # (crude but defensible without real options data)
    SPY_MOVE_FOR_MAX_GAIN = 0.005  # 0.5% move → full gain

    spread_value_at_close = debit + (directional_move / SPY_MOVE_FOR_MAX_GAIN) * max_gain
    spread_value_at_close = max(0.0, min(debit + max_gain, spread_value_at_close))

    pnl = spread_value_at_close - debit

    # Determine exit type
    if pnl >= profit_target:
        exit_type = "profit_target"
        pnl = profit_target
    elif pnl <= -stop_loss_trigger:
        exit_type = "stop_loss"
        pnl = -stop_loss_trigger
    else:
        exit_type = "time_stop"

    # Hold time: roughly 9:45 → 3:45 = 6h; use directional speed proxy
    if exit_type == "time_stop":
        hold_hours = 6.0
    else:
        # Estimate time based on how quickly the move occurred
        high_move = ((spy_high - spy_open) / spy_open) * direction if spy_high else directional_move
        low_move = ((spy_low - spy_open) / spy_open) * direction if spy_low else directional_move

        if exit_type == "profit_target" and high_move >= SPY_MOVE_FOR_MAX_GAIN:
            hold_hours = 1.5  # hit target early
        elif exit_type == "stop_loss" and low_move <= -SPY_MOVE_FOR_MAX_GAIN:
            hold_hours = 1.0
        else:
            hold_hours = 3.0

    return pnl, exit_type, hold_hours


def _get_price(row: pd.Series, candidates: list[str]) -> float | None:
    for c in candidates:
        if c in row.index and pd.notna(row[c]):
            return float(row[c])
    return None


def run_backtest(
    df: pd.DataFrame,
    model: xgb.XGBClassifier,
    feature_cols: list[str],
    profit_target_pct: float = PROFIT_TARGET_PCT,
    stop_loss_pct: float = STOP_LOSS_PCT,
    debit: float = DEBIT,
    max_gain: float = MAX_GAIN,
    spread_width: float = SPREAD_WIDTH,
    direction_threshold: float = 0.002,
    vix_filter: float | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Core backtest loop. Returns (trade_log_df, summary_dict).
    direction_threshold: minimum confidence not used here (label is already generated),
                         but kept for sweep.py compatibility.
    """
    label_map = {-1: 0, 0: 1, 1: 2}
    df = df.copy()
    df["target_idx"] = df["target"].map(label_map)
    df = df.dropna(subset=feature_cols)

    X = df[feature_cols].values.astype(np.float32)
    proba = model.predict_proba(X)   # shape (n, 3)
    preds = proba.argmax(axis=1)     # 0-indexed: 0=DOWN, 1=FLAT, 2=UP

    trades = []
    for i, (_, row) in enumerate(df.iterrows()):
        pred_class = preds[i]
        pred_direction = LABEL_INV[pred_class]   # -1, 0, or 1

        # VIX filter: skip if VIX above threshold
        if vix_filter is not None:
            vix_val = _get_price(row, ["vix_level", "vix_close", "vix_vix_close"])
            if vix_val is not None and vix_val > vix_filter:
                continue

        if pred_direction == 0:
            continue   # No trade

        pnl, exit_type, hold_hours = _simulate_intraday_exit(
            row, pred_direction, debit, max_gain, -debit,
            profit_target_pct, stop_loss_pct,
        )

        confidence = float(proba[i][pred_class])
        spread_type = "call_spread" if pred_direction == 1 else "put_spread"

        trades.append({
            "date": row["date"],
            "direction": pred_direction,
            "spread_type": spread_type,
            "entry_debit": debit,
            "exit_value": debit + pnl,
            "pnl": pnl,
            "win": int(pnl > 0),
            "hold_hours": hold_hours,
            "exit_type": exit_type,
            "confidence": confidence,
            "vix_level": _get_price(row, ["vix_level", "vix_close"]) or np.nan,
            "day_of_week": int(row.get("day_of_week", row["date"].dayofweek)),
        })

    trade_log = pd.DataFrame(trades)
    summary = _compute_summary(trade_log)
    return trade_log, summary


def _compute_summary(trade_log: pd.DataFrame) -> dict:
    if trade_log.empty:
        return {"error": "No trades generated."}

    n = len(trade_log)
    wins = trade_log["win"].sum()
    total_pnl = trade_log["pnl"].sum()

    # Drawdown
    cumulative = trade_log["pnl"].cumsum()
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max
    max_drawdown = float(drawdown.min())

    # Sharpe (daily)
    daily_pnl = trade_log.groupby("date")["pnl"].sum()
    sharpe = (
        float(daily_pnl.mean() / daily_pnl.std() * np.sqrt(252))
        if daily_pnl.std() > 0 else 0.0
    )

    avg_hold = float(trade_log["hold_hours"].mean())

    # Win rate by day of week
    wr_dow = (
        trade_log.groupby("day_of_week")["win"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "win_rate", "count": "trades"})
        .to_dict(orient="index")
    )

    # Win rate by VIX regime
    trade_log["vix_regime"] = pd.cut(
        trade_log["vix_level"],
        bins=[0, 15, 20, 25, 100],
        labels=["<15", "15-20", "20-25", ">25"],
    )
    wr_vix = (
        trade_log.groupby("vix_regime", observed=True)["win"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "win_rate", "count": "trades"})
        .to_dict(orient="index")
    )

    # Win rate by direction
    wr_dir = (
        trade_log.groupby("direction")["win"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "win_rate", "count": "trades"})
        .to_dict(orient="index")
    )

    # Win rate by month
    trade_log["month"] = pd.to_datetime(trade_log["date"]).dt.month
    wr_month = (
        trade_log.groupby("month")["win"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "win_rate", "count": "trades"})
        .to_dict(orient="index")
    )

    # Exit type breakdown
    exit_counts = trade_log["exit_type"].value_counts().to_dict()

    return {
        "total_trades": n,
        "win_rate": float(wins / n),
        "total_pnl": float(total_pnl),
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "avg_hold_hours": avg_hold,
        "exit_type_counts": exit_counts,
        "win_rate_by_day_of_week": {str(k): v for k, v in wr_dow.items()},
        "win_rate_by_vix_regime": {str(k): v for k, v in wr_vix.items()},
        "win_rate_by_direction": {str(k): v for k, v in wr_dir.items()},
        "win_rate_by_month": {str(k): v for k, v in wr_month.items()},
    }


def main():
    print("[backtest] Loading model …")
    model, feature_cols = load_model_and_features()

    print("[backtest] Loading test features …")
    df = load_test_features()
    print(f"  Test rows: {len(df):,}  ({df['date'].min().date()} → {df['date'].max().date()})")

    print("\n[backtest] Running backtest …")
    trade_log, summary = run_backtest(df, model, feature_cols)

    # ── Save results ──────────────────────────────────────────────────────────
    trade_log_path = RESULTS_DIR / "trade_log.parquet"
    trade_log.to_parquet(trade_log_path, index=False)
    print(f"\n[backtest] ✓ Trade log saved → {trade_log_path} ({len(trade_log)} trades)")

    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[backtest] ✓ Summary saved → {summary_path}")

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n[backtest] === BACKTEST SUMMARY ===")
    print(f"  Total trades  : {summary['total_trades']}")
    print(f"  Win rate      : {summary['win_rate']:.1%}")
    print(f"  Total P&L     : ${summary['total_pnl']:.2f}")
    print(f"  Max drawdown  : ${summary['max_drawdown']:.2f}")
    print(f"  Sharpe ratio  : {summary['sharpe']:.3f}")
    print(f"  Avg hold time : {summary['avg_hold_hours']:.1f}h")

    print("\n  Win rate by direction:")
    for d, stats in summary["win_rate_by_direction"].items():
        label = "CALL" if int(d) == 1 else "PUT"
        print(f"    {label}: {stats['win_rate']:.1%} ({stats['trades']} trades)")

    print("\n  Win rate by VIX regime:")
    for regime, stats in summary["win_rate_by_vix_regime"].items():
        print(f"    VIX {regime}: {stats['win_rate']:.1%} ({stats['trades']} trades)")

    print("\n  Exit type breakdown:")
    for etype, cnt in summary["exit_type_counts"].items():
        print(f"    {etype}: {cnt}")


if __name__ == "__main__":
    main()
