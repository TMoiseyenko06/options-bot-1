"""
grid_search/sweep.py — Sweep parameter combinations over the backtest engine.

Parameters swept:
  direction_threshold : [0.1%, 0.2%, 0.3%]
  spread_width        : [2, 3, 5 points]
  profit_target       : [50%, 75%, 100%]
  stop_loss           : [50%, 75%, 100%]
  vix_filter          : [13, 15, 17] (max VIX to trade)

Saves grid_search/results/sweep.parquet and Plotly heatmaps.
"""

import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backtest.engine import run_backtest, load_model_and_features, load_test_features

RESULTS_DIR = PROJECT_ROOT / "grid_search" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SWEEP_PARAMS = {
    "direction_threshold": [0.001, 0.002, 0.003],   # 0.1%, 0.2%, 0.3%
    "spread_width": [2, 3, 5],
    "profit_target_pct": [0.50, 0.75, 1.00],
    "stop_loss_pct": [0.50, 0.75, 1.00],
    "vix_filter": [13, 15, 17],
}

ENTRY_TIMES = ["09:30", "09:45", "10:00"]  # stored as metadata; engine uses 9:45 internally


def _params_to_debit_and_gain(spread_width: int, debit_ratio: float = 0.35):
    debit = spread_width * debit_ratio * 100
    max_gain = spread_width * 100 - debit
    return debit, max_gain


def run_sweep(df: pd.DataFrame, model: xgb.XGBClassifier, feature_cols: list[str]):
    keys = list(SWEEP_PARAMS.keys())
    combos = list(product(*[SWEEP_PARAMS[k] for k in keys]))

    # Add entry_time axis (metadata only — engine doesn't model exact entry time)
    all_results = []
    total = len(combos) * len(ENTRY_TIMES)
    print(f"[sweep] Running {total} combinations …")

    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        debit, max_gain = _params_to_debit_and_gain(int(params["spread_width"]))

        for entry_time in ENTRY_TIMES:
            try:
                _, summary = run_backtest(
                    df,
                    model,
                    feature_cols,
                    profit_target_pct=params["profit_target_pct"],
                    stop_loss_pct=params["stop_loss_pct"],
                    debit=debit,
                    max_gain=max_gain,
                    spread_width=params["spread_width"],
                    direction_threshold=params["direction_threshold"],
                    vix_filter=params["vix_filter"],
                )

                if "error" in summary:
                    continue

                row = {
                    "direction_threshold": params["direction_threshold"],
                    "spread_width": params["spread_width"],
                    "profit_target_pct": params["profit_target_pct"],
                    "stop_loss_pct": params["stop_loss_pct"],
                    "vix_filter": params["vix_filter"],
                    "entry_time": entry_time,
                    "total_trades": summary["total_trades"],
                    "win_rate": summary["win_rate"],
                    "total_pnl": summary["total_pnl"],
                    "max_drawdown": summary["max_drawdown"],
                    "sharpe": summary["sharpe"],
                    "avg_hold_hours": summary["avg_hold_hours"],
                }
                all_results.append(row)

            except Exception as e:
                print(f"  ERROR at combo {i+1}: {e}")

        if (i + 1) % 50 == 0:
            print(f"  … {i+1}/{len(combos)} combos done")

    results_df = pd.DataFrame(all_results)
    return results_df


def _pivot_heatmap(
    df: pd.DataFrame,
    row_col: str,
    col_col: str,
    value_col: str,
    title: str,
    colorscale: str = "RdYlGn",
) -> go.Figure:
    """Create a Plotly heatmap from a sweep results DataFrame."""
    pivot = df.groupby([row_col, col_col])[value_col].mean().unstack(col_col)

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[str(c) for c in pivot.columns],
        y=[str(r) for r in pivot.index],
        colorscale=colorscale,
        text=np.round(pivot.values, 3),
        texttemplate="%{text}",
        showscale=True,
        colorbar=dict(title=value_col),
    ))
    fig.update_layout(
        title=title,
        xaxis_title=col_col,
        yaxis_title=row_col,
        height=500,
        width=700,
    )
    return fig


def _stable_region_heatmap(
    df: pd.DataFrame,
    row_col: str,
    col_col: str,
    metric: str = "sharpe",
    title: str = "Stability (low std = stable region)",
) -> go.Figure:
    """Heatmap of std dev across other parameters — low std means stable region."""
    std_pivot = df.groupby([row_col, col_col])[metric].std().unstack(col_col)

    fig = go.Figure(go.Heatmap(
        z=std_pivot.values,
        x=[str(c) for c in std_pivot.columns],
        y=[str(r) for r in std_pivot.index],
        colorscale="YlOrRd_r",   # reversed: green=stable
        text=np.round(std_pivot.values, 3),
        texttemplate="%{text}",
        showscale=True,
        colorbar=dict(title=f"std({metric})"),
    ))
    fig.update_layout(
        title=title,
        xaxis_title=col_col,
        yaxis_title=row_col,
        height=500,
        width=700,
    )
    return fig


def generate_heatmaps(results_df: pd.DataFrame):
    """Generate and save all heatmaps."""
    pairs = [
        ("profit_target_pct", "stop_loss_pct"),
        ("spread_width", "vix_filter"),
        ("profit_target_pct", "spread_width"),
        ("stop_loss_pct", "vix_filter"),
    ]

    for row_col, col_col in pairs:
        # Win rate heatmap
        fig = _pivot_heatmap(
            results_df, row_col, col_col, "win_rate",
            f"Win Rate: {row_col} vs {col_col}",
        )
        fname = f"heatmap_winrate_{row_col}_vs_{col_col}.html"
        fig.write_html(str(RESULTS_DIR / fname))
        print(f"[sweep] Saved → {fname}")

        # Sharpe heatmap
        fig = _pivot_heatmap(
            results_df, row_col, col_col, "sharpe",
            f"Sharpe Ratio: {row_col} vs {col_col}",
        )
        fname = f"heatmap_sharpe_{row_col}_vs_{col_col}.html"
        fig.write_html(str(RESULTS_DIR / fname))
        print(f"[sweep] Saved → {fname}")

        # Stability (low variance) heatmap
        fig = _stable_region_heatmap(
            results_df, row_col, col_col, "sharpe",
            f"Sharpe Stability (low std = robust): {row_col} vs {col_col}",
        )
        fname = f"heatmap_stability_{row_col}_vs_{col_col}.html"
        fig.write_html(str(RESULTS_DIR / fname))
        print(f"[sweep] Saved → {fname}")


def main():
    print("[sweep] Loading model and test data …")
    model, feature_cols = load_model_and_features()
    df = load_test_features()
    print(f"  Test rows: {len(df):,}")

    results_df = run_sweep(df, model, feature_cols)

    # Save raw results
    out_path = RESULTS_DIR / "sweep.parquet"
    results_df.to_parquet(out_path, index=False)
    print(f"\n[sweep] ✓ Sweep results saved → {out_path} ({len(results_df):,} rows)")

    # Print top 10 by Sharpe
    print("\n[sweep] Top 10 combinations by Sharpe ratio:")
    top = results_df.nlargest(10, "sharpe")
    print(top[[
        "spread_width", "profit_target_pct", "stop_loss_pct",
        "vix_filter", "win_rate", "sharpe", "total_pnl"
    ]].to_string(index=False))

    print("\n[sweep] Generating heatmaps …")
    generate_heatmaps(results_df)

    print("\n[sweep] Done.")


if __name__ == "__main__":
    main()
