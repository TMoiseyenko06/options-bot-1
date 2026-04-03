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


def load_spy_intraday() -> pd.DataFrame | None:
    """
    Load 1-minute SPY intraday bars if available.
    Returns DataFrame with columns [ts_et, date, open, high, low, close]
    or None if the file does not exist.
    """
    path = RAW_DIR / "spy_intraday_1m.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["ts_et"] = pd.to_datetime(df["ts_et"])
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("ts_et").reset_index(drop=True)


# SPY move required for a spread to reach max gain (call/put).
# A 0.5% directional move in the underlying roughly puts a 3-point
# ATM debit spread fully in-the-money for 0DTE.
SPY_MOVE_FOR_MAX_GAIN = 0.005


def _spread_pnl_from_move(directional_move: float, debit: float, max_gain: float) -> float:
    """
    Map a directional SPY move (positive = favourable) to spread P&L dollars.
    Linear interpolation between 0 (no move) and max_gain (full move).
    Clipped at [−debit, max_gain].
    """
    raw = (directional_move / SPY_MOVE_FOR_MAX_GAIN) * max_gain
    return float(np.clip(raw, -debit, max_gain))


def _simulate_exit_minute_bars(
    date_bars: pd.DataFrame,
    entry_price: float,
    direction: int,
    debit: float,
    max_gain: float,
    profit_target_pct: float,
    stop_loss_pct: float,
) -> tuple[float, str, float]:
    """
    Walk 1-minute bars from 9:45am to 3:45pm ET.
    On each bar check whether the high (favourable) or low (adverse) touches
    the profit target or stop loss before the other is checked.

    Convention: for a call spread, SPY rising is favourable (check high first),
    falling is adverse (check low).  Reversed for put spreads.

    Returns (pnl_dollars, exit_type, hold_hours_from_entry).
    """
    profit_target_dollars = debit * profit_target_pct
    stop_dollars = debit * stop_loss_pct

    # Entry is at the open of the 9:45 bar
    entry_time_minutes = 9 * 60 + 45
    time_stop_minutes = 15 * 60 + 45   # 3:45pm

    date_bars = date_bars.copy()
    date_bars["bar_minutes"] = (
        date_bars["ts_et"].dt.hour * 60 + date_bars["ts_et"].dt.minute
    )

    trade_bars = date_bars[
        (date_bars["bar_minutes"] >= entry_time_minutes) &
        (date_bars["bar_minutes"] <= time_stop_minutes)
    ].sort_values("bar_minutes")

    if trade_bars.empty:
        return 0.0, "no_data", 0.0

    for _, bar in trade_bars.iterrows():
        bar_high = float(bar["high"])
        bar_low = float(bar["low"])

        # Favourable and adverse prices depend on direction
        if direction == 1:   # call spread: high is good, low is bad
            fav_move = (bar_high - entry_price) / entry_price
            adv_move = (bar_low - entry_price) / entry_price
        else:                 # put spread: low is good, high is bad
            fav_move = (entry_price - bar_low) / entry_price
            adv_move = (entry_price - bar_high) / entry_price

        fav_pnl = _spread_pnl_from_move(fav_move, debit, max_gain)
        adv_pnl = _spread_pnl_from_move(adv_move, debit, max_gain)

        hold_hours = (bar["bar_minutes"] - entry_time_minutes) / 60.0

        # Within a single bar assume favourable move is checked before adverse
        # (conservative: assume best-case for profit target, worst-case for stop)
        if fav_pnl >= profit_target_dollars:
            return profit_target_dollars, "profit_target", hold_hours
        if adv_pnl <= -stop_dollars:
            return -stop_dollars, "stop_loss", hold_hours

    # Time stop: use close of last bar
    last_bar = trade_bars.iloc[-1]
    close_price = float(last_bar["close"])
    close_move = (close_price - entry_price) / entry_price * direction
    pnl = _spread_pnl_from_move(close_move, debit, max_gain)
    # Clip to valid range (don't exceed targets at time stop)
    pnl = float(np.clip(pnl, -stop_dollars, profit_target_dollars))
    hold_hours = (last_bar["bar_minutes"] - entry_time_minutes) / 60.0
    return pnl, "time_stop", hold_hours


def _simulate_exit_daily_fallback(
    row: pd.Series,
    direction: int,
    debit: float,
    max_gain: float,
    profit_target_pct: float,
    stop_loss_pct: float,
) -> tuple[float, str, float]:
    """
    Fallback when no 1-minute data is available.
    Uses daily high/low to check if target or stop was touched,
    and close to determine time-stop P&L.

    Limitation: cannot determine intraday order of high vs low.
    We assume: if both target and stop would have been hit, stop takes priority
    (conservative assumption — avoids overstating win rate).
    """
    spy_open = _get_price(row, ["spy_open"])
    spy_close = _get_price(row, ["spy_close", "spy_adj_close"])
    spy_high = _get_price(row, ["spy_high"])
    spy_low = _get_price(row, ["spy_low"])

    if spy_open is None or spy_close is None:
        return 0.0, "no_data", 0.0

    profit_target_dollars = debit * profit_target_pct
    stop_dollars = debit * stop_loss_pct

    if direction == 1:
        fav_extreme = spy_high
        adv_extreme = spy_low
    else:
        fav_extreme = spy_low
        adv_extreme = spy_high

    fav_move = abs(fav_extreme - spy_open) / spy_open if fav_extreme else 0.0
    adv_move = abs(adv_extreme - spy_open) / spy_open if adv_extreme else 0.0

    # Flip sign: adv move is negative
    fav_pnl = _spread_pnl_from_move(fav_move, debit, max_gain)
    adv_pnl = _spread_pnl_from_move(-adv_move, debit, max_gain)

    target_hit = fav_pnl >= profit_target_dollars
    stop_hit = adv_pnl <= -stop_dollars

    if stop_hit:
        # Conservative: if stop could have been hit, assume it was
        return -stop_dollars, "stop_loss", 2.0
    if target_hit:
        return profit_target_dollars, "profit_target", 2.0

    # Neither hit — use close
    close_move = (spy_close - spy_open) / spy_open * direction
    pnl = _spread_pnl_from_move(close_move, debit, max_gain)
    pnl = float(np.clip(pnl, -stop_dollars, profit_target_dollars))
    return pnl, "time_stop", 6.0


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
    intraday_df: pd.DataFrame | None = None,
    options_pricing: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Core backtest loop. Returns (trade_log_df, summary_dict).

    intraday_df: 1-minute SPY bars (from spy_intraday_1m.parquet).
                 If provided, exits are simulated bar-by-bar.
                 If None, falls back to daily OHLCV approximation.

    options_pricing: lookup dict from backtest.historical_options.load_options_cache().
                     Keys are (date_str, contract_type, spread_width).
                     When a key is found, the actual API debit/max_gain replaces the
                     fixed 35% approximation for that trade.
                     If None or key missing, falls back to the parametric debit/max_gain.
    """
    label_map = {-1: 0, 0: 1, 1: 2}
    df = df.copy()
    df["target_idx"] = df["target"].map(label_map)
    df = df.dropna(subset=feature_cols)

    # Pre-index intraday data by date for fast lookup
    intraday_by_date: dict[pd.Timestamp, pd.DataFrame] = {}
    if intraday_df is not None and not intraday_df.empty:
        for trade_date, group in intraday_df.groupby("date"):
            intraday_by_date[pd.Timestamp(trade_date)] = group
        print(f"[backtest] Using 1-min intraday exit simulation ({len(intraday_by_date)} trading days)")
    else:
        print("[backtest] No intraday data found — using daily OHLCV fallback for exits")

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

        trade_date = pd.Timestamp(row["date"]).normalize()

        # ── Real options pricing (15-min delayed, from Massive API cache) ─────
        # Look up actual debit and max_gain for this date/direction/width.
        # Falls back to the parametric 35% approximation when not in cache.
        contract_type = "call" if pred_direction == 1 else "put"
        pricing_source = "35pct_approx"
        trade_debit = debit
        trade_max_gain = max_gain
        if options_pricing is not None:
            cache_key = (str(trade_date.date()), contract_type, int(spread_width))
            cached = options_pricing.get(cache_key)
            if cached is not None:
                trade_debit = cached["debit_dollars"]
                trade_max_gain = cached["max_gain_dollars"]
                pricing_source = "api_real"

        day_bars = intraday_by_date.get(trade_date)

        if day_bars is not None and not day_bars.empty:
            # Get entry price: open of the 9:45am bar
            entry_mask = (
                day_bars["ts_et"].dt.hour * 60 + day_bars["ts_et"].dt.minute == 9 * 60 + 45
            )
            entry_bars = day_bars[entry_mask]
            if entry_bars.empty:
                # Fall back to first bar at or after 9:45
                after_945 = day_bars[
                    day_bars["ts_et"].dt.hour * 60 + day_bars["ts_et"].dt.minute >= 9 * 60 + 45
                ]
                entry_price = float(after_945.iloc[0]["open"]) if not after_945.empty else None
            else:
                entry_price = float(entry_bars.iloc[0]["open"])

            if entry_price is None:
                continue

            pnl, exit_type, hold_hours = _simulate_exit_minute_bars(
                day_bars, entry_price, pred_direction,
                trade_debit, trade_max_gain, profit_target_pct, stop_loss_pct,
            )
        else:
            # Daily fallback
            pnl, exit_type, hold_hours = _simulate_exit_daily_fallback(
                row, pred_direction, trade_debit, trade_max_gain,
                profit_target_pct, stop_loss_pct,
            )

        confidence = float(proba[i][pred_class])
        spread_type = "call_spread" if pred_direction == 1 else "put_spread"

        trades.append({
            "date": row["date"],
            "direction": pred_direction,
            "spread_type": spread_type,
            "entry_debit": trade_debit,
            "exit_value": trade_debit + pnl,
            "pnl": pnl,
            "win": int(pnl > 0),
            "hold_hours": hold_hours,
            "exit_type": exit_type,
            "confidence": confidence,
            "vix_level": _get_price(row, ["vix_level", "vix_close"]) or np.nan,
            "day_of_week": int(row.get("day_of_week", row["date"].dayofweek)),
            "pricing_source": pricing_source,
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

    # Options pricing source breakdown (api_real vs 35pct_approx)
    pricing_source_counts = {}
    if "pricing_source" in trade_log.columns:
        pricing_source_counts = trade_log["pricing_source"].value_counts().to_dict()

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
        "pricing_source_counts": pricing_source_counts,
    }


def main():
    print("[backtest] Loading model …")
    model, feature_cols = load_model_and_features()

    print("[backtest] Loading test features …")
    df = load_test_features()
    print(f"  Test rows: {len(df):,}  ({df['date'].min().date()} → {df['date'].max().date()})")

    print("[backtest] Loading intraday data …")
    intraday_df = load_spy_intraday()
    if intraday_df is not None:
        # Filter to test period only
        intraday_df = intraday_df[intraday_df["date"] >= TEST_START]
        print(f"  Intraday bars: {len(intraday_df):,} (1-min, market hours)")
    else:
        print("  No spy_intraday_1m.parquet found — will use daily OHLCV fallback")

    # ── Real options pricing via Massive API (15-min delayed, consistent with live) ──
    print("[backtest] Loading real options pricing from Massive API …")
    from backtest.historical_options import load_options_cache
    options_pricing = load_options_cache(df, spread_widths=[SPREAD_WIDTH])

    print("\n[backtest] Running backtest …")
    trade_log, summary = run_backtest(
        df, model, feature_cols,
        intraday_df=intraday_df,
        options_pricing=options_pricing,
    )

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

    if "pricing_source_counts" in summary:
        print("\n  Options pricing source:")
        for src, cnt in summary["pricing_source_counts"].items():
            print(f"    {src}: {cnt}")


if __name__ == "__main__":
    main()
