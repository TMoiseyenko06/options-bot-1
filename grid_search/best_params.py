"""
grid_search/best_params.py — Extract the best parameter combination from sweep results
and run a detailed backtest with it.

Usage:
    python grid_search/best_params.py
    python grid_search/best_params.py --metric win_rate
    python grid_search/best_params.py --metric total_pnl
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backtest.engine import (
    run_backtest,
    load_model_and_features,
    load_spy_intraday,
    load_test_features,
)
from grid_search.sweep import _params_to_debit_and_gain

SWEEP_PATH = PROJECT_ROOT / "grid_search" / "results" / "sweep.parquet"
RESULTS_DIR = PROJECT_ROOT / "backtest" / "results"

TEST_START = "2023-01-01"


def load_sweep() -> pd.DataFrame:
    if not SWEEP_PATH.exists():
        raise FileNotFoundError(f"No sweep results found at {SWEEP_PATH}\nRun: python grid_search/sweep.py")
    return pd.read_parquet(SWEEP_PATH)


def print_best(best: pd.Series, metric: str):
    print(f"\n{'='*55}")
    print(f"  BEST PARAMETERS  (ranked by {metric})")
    print(f"{'='*55}")
    print(f"  spread_width       : {int(best['spread_width'])} points")
    print(f"  profit_target      : {best['profit_target_pct']:.0%}")
    print(f"  stop_loss          : {best['stop_loss_pct']:.0%}")
    print(f"  vix_filter         : VIX < {best['vix_filter']}")
    print(f"  direction_threshold: {best['direction_threshold']:.1%}")
    print(f"\n  Sweep metrics:")
    print(f"    win_rate   : {best['win_rate']:.1%}")
    print(f"    sharpe     : {best['sharpe']:.3f}")
    print(f"    total_pnl  : ${best['total_pnl']:.2f}")
    print(f"    max_dd     : ${best['max_drawdown']:.2f}")
    print(f"    trades     : {int(best['total_trades'])}")
    print(f"{'='*55}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metric",
        default="combined",
        choices=["sharpe", "win_rate", "total_pnl", "combined"],
        help="Metric to rank by. 'combined' maximises win_rate x total_pnl (default)",
    )
    args = parser.parse_args()

    print(f"[best_params] Loading sweep results …")
    sweep = load_sweep()
    print(f"  {len(sweep):,} combinations found")

    # Combined score: normalise win_rate and total_pnl to [0,1] then multiply
    # This rewards combos that are strong on both axes, not just one
    if args.metric == "combined":
        sweep["_wr_norm"] = (sweep["win_rate"] - sweep["win_rate"].min()) / (
            sweep["win_rate"].max() - sweep["win_rate"].min() + 1e-9
        )
        sweep["_pnl_norm"] = (sweep["total_pnl"] - sweep["total_pnl"].min()) / (
            sweep["total_pnl"].max() - sweep["total_pnl"].min() + 1e-9
        )
        sweep["combined_score"] = sweep["_wr_norm"] * sweep["_pnl_norm"]
        rank_col = "combined_score"
        label = "combined (win_rate x total_pnl)"
    else:
        rank_col = args.metric
        label = args.metric

    best = sweep.nlargest(1, rank_col).iloc[0]
    print_best(best, label)

    # Top 10 for context — sorted by trade count so you can see the "more trades" options too
    print(f"Top 10 by {label}  (sorted by trade count descending):")
    cols = ["spread_width", "profit_target_pct", "stop_loss_pct",
            "vix_filter", "total_trades", "win_rate", "sharpe", "total_pnl"]
    top10 = sweep.nlargest(10, rank_col)[cols].sort_values("total_trades", ascending=False)
    print(top10.to_string(index=False))

    print(f"\n[best_params] Running detailed backtest with best params …")
    model, feature_cols = load_model_and_features()
    df = load_test_features()

    intraday_df = load_spy_intraday()
    if intraday_df is not None:
        intraday_df = intraday_df[intraday_df["date"] >= TEST_START]

    spread_width = int(best["spread_width"])
    debit = spread_width * 0.35 * 100
    max_gain = spread_width * 100 - debit

    # Load real options pricing (15-min delayed) for the best spread width
    from backtest.historical_options import load_options_cache
    options_pricing = load_options_cache(df, spread_widths=[spread_width])

    trade_log, summary = run_backtest(
        df,
        model,
        feature_cols,
        profit_target_pct=float(best["profit_target_pct"]),
        stop_loss_pct=float(best["stop_loss_pct"]),
        debit=debit,
        max_gain=max_gain,
        spread_width=spread_width,
        direction_threshold=float(best["direction_threshold"]),
        vix_filter=float(best["vix_filter"]),
        intraday_df=intraday_df,
        options_pricing=options_pricing,
    )

    print(f"\n{'='*55}")
    print(f"  DETAILED BACKTEST RESULTS (2023-present)")
    print(f"{'='*55}")
    print(f"  Total trades   : {summary['total_trades']}")
    print(f"  Win rate       : {summary['win_rate']:.1%}")
    print(f"  Total P&L      : ${summary['total_pnl']:.2f}")
    print(f"  Max drawdown   : ${summary['max_drawdown']:.2f}")
    print(f"  Sharpe ratio   : {summary['sharpe']:.3f}")
    print(f"  Avg hold time  : {summary['avg_hold_hours']:.1f}h")

    print(f"\n  Win rate by direction:")
    for d, s in summary["win_rate_by_direction"].items():
        label = "CALL" if int(d) == 1 else "PUT"
        print(f"    {label}: {s['win_rate']:.1%}  ({s['trades']} trades)")

    print(f"\n  Win rate by VIX regime:")
    for regime, s in summary["win_rate_by_vix_regime"].items():
        print(f"    VIX {regime}: {s['win_rate']:.1%}  ({s['trades']} trades)")

    print(f"\n  Exit type breakdown:")
    for etype, cnt in summary["exit_type_counts"].items():
        print(f"    {etype}: {cnt}")

    if summary.get("pricing_source_counts"):
        print(f"\n  Options pricing source:")
        for src, cnt in summary["pricing_source_counts"].items():
            print(f"    {src}: {cnt}")

    # Save best params for reference
    best_params_out = {
        "spread_width": spread_width,
        "profit_target_pct": float(best["profit_target_pct"]),
        "stop_loss_pct": float(best["stop_loss_pct"]),
        "vix_filter": float(best["vix_filter"]),
        "direction_threshold": float(best["direction_threshold"]),
        "ranked_by": args.metric,
        "sweep_sharpe": float(best["sharpe"]),
        "sweep_win_rate": float(best["win_rate"]),
    }
    out_path = PROJECT_ROOT / "grid_search" / "results" / "best_params.json"
    with open(out_path, "w") as f:
        json.dump(best_params_out, f, indent=2)
    print(f"\n[best_params] Best params saved -> {out_path}")

    # Save trade log
    tl_path = RESULTS_DIR / "best_trade_log.parquet"
    trade_log.to_parquet(tl_path, index=False)
    print(f"[best_params] Trade log saved -> {tl_path}")


if __name__ == "__main__":
    main()
