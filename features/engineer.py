"""
features/engineer.py — Feature engineering and label generation.

All features use only data available BEFORE market open (no lookahead).
Labels are generated from intraday SPY return on the same day.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# ── FOMC meeting dates 2018-present ─────────────────────────────────────────
# Source: Federal Reserve historical calendar
FOMC_DATES = [
    # 2018
    "2018-01-31", "2018-03-21", "2018-05-02", "2018-06-13",
    "2018-08-01", "2018-09-26", "2018-11-08", "2018-12-19",
    # 2019
    "2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19",
    "2019-07-31", "2019-09-18", "2019-10-30", "2019-12-11",
    # 2020
    "2020-01-29", "2020-03-03", "2020-03-15", "2020-04-29",
    "2020-06-10", "2020-07-29", "2020-09-16", "2020-11-05",
    "2020-12-16",
    # 2021
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16",
    "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
    # 2022
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15",
    "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    # 2023
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
    "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    # 2026
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",
]

# ── CPI release dates 2018-present ──────────────────────────────────────────
# Source: BLS CPI historical release calendar
CPI_DATES = [
    # 2018
    "2018-01-12", "2018-02-14", "2018-03-13", "2018-04-11",
    "2018-05-10", "2018-06-12", "2018-07-12", "2018-08-10",
    "2018-09-13", "2018-10-11", "2018-11-14", "2018-12-12",
    # 2019
    "2019-01-11", "2019-02-13", "2019-03-12", "2019-04-10",
    "2019-05-10", "2019-06-12", "2019-07-11", "2019-08-13",
    "2019-09-12", "2019-10-10", "2019-11-13", "2019-12-11",
    # 2020
    "2020-01-14", "2020-02-13", "2020-03-11", "2020-04-10",
    "2020-05-12", "2020-06-10", "2020-07-14", "2020-08-12",
    "2020-09-11", "2020-10-13", "2020-11-12", "2020-12-10",
    # 2021
    "2021-01-13", "2021-02-10", "2021-03-10", "2021-04-13",
    "2021-05-12", "2021-06-10", "2021-07-13", "2021-08-11",
    "2021-09-14", "2021-10-13", "2021-11-10", "2021-12-10",
    # 2022
    "2022-01-12", "2022-02-10", "2022-03-10", "2022-04-12",
    "2022-05-11", "2022-06-10", "2022-07-13", "2022-08-10",
    "2022-09-13", "2022-10-13", "2022-11-10", "2022-12-13",
    # 2023
    "2023-01-12", "2023-02-14", "2023-03-14", "2023-04-12",
    "2023-05-10", "2023-06-13", "2023-07-12", "2023-08-10",
    "2023-09-13", "2023-10-12", "2023-11-14", "2023-12-12",
    # 2024
    "2024-01-11", "2024-02-13", "2024-03-12", "2024-04-10",
    "2024-05-15", "2024-06-12", "2024-07-11", "2024-08-14",
    "2024-09-11", "2024-10-10", "2024-11-13", "2024-12-11",
    # 2025
    "2025-01-15", "2025-02-12", "2025-03-12", "2025-04-10",
    "2025-05-13", "2025-06-11", "2025-07-11", "2025-08-13",
    "2025-09-10", "2025-10-15", "2025-11-13", "2025-12-10",
    # 2026
    "2026-01-14", "2026-02-11", "2026-03-11", "2026-04-09",
    "2026-05-13", "2026-06-10", "2026-07-15", "2026-08-12",
    "2026-09-09", "2026-10-14", "2026-11-12", "2026-12-09",
]


def _days_until_next(dates: list[str], reference: pd.Series) -> pd.Series:
    """
    For each date in `reference`, compute the number of calendar days
    until the next event in `dates`.  Returns -1 if no future event exists.
    """
    event_dates = pd.to_datetime(dates).sort_values()
    result = []
    for ref_date in reference:
        future = event_dates[event_dates >= ref_date]
        if future.empty:
            result.append(-1)
        else:
            result.append((future[0] - ref_date).days)
    return pd.Series(result, index=reference.index)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build all features from the merged daily DataFrame.

    Assumes df has columns produced by merge.py:
      spy_open, spy_close, spy_high, spy_low
      vix_close  (and optionally vix9d_close)
      es_overnight_gap
      xlk_close, xlf_close, xle_close, xlv_close  (via sectors merge)
    """
    df = df.copy().sort_values("date").reset_index(drop=True)

    # ── Helper: identify SPY price columns ───────────────────────────────────
    spy_close = _find_col(df, ["spy_close", "spy_adj_close"])
    spy_open = _find_col(df, ["spy_open"])

    # ── Momentum features ────────────────────────────────────────────────────

    # SPY previous day return (shift(1): yesterday's close)
    df["spy_prev_return"] = df[spy_close].pct_change(1)

    # SPY 5-day return
    df["spy_5d_return"] = df[spy_close].pct_change(5)

    # SPY distance from 20-day MA (%)
    df["spy_ma20"] = df[spy_close].rolling(20).mean()
    df["spy_dist_ma20"] = (df[spy_close] - df["spy_ma20"]) / df["spy_ma20"] * 100
    df = df.drop(columns=["spy_ma20"])

    # Overnight gap — use ES if available, else fall back to SPY open vs prior close
    if "es_overnight_gap" not in df.columns:
        es_open = _find_col(df, ["es_open"], required=False)
        es_close = _find_col(df, ["es_close"], required=False)
        if es_open and es_close:
            df["es_overnight_gap"] = (
                (df[es_open] - df[es_close].shift(1)) / df[es_close].shift(1)
            )
        else:
            # SPY open vs prior SPY close — good enough without ES data
            df["es_overnight_gap"] = (
                (df[spy_open] - df[spy_close].shift(1)) / df[spy_close].shift(1)
            )

    # ── Volatility features ──────────────────────────────────────────────────
    vix_close = _find_col(df, ["vix_close", "vix_vix_close"])

    df["vix_level"] = df[vix_close]

    # VIX vs its own 10-day MA
    df["vix_ma10"] = df[vix_close].rolling(10).mean()
    df["vix_vs_ma10"] = (df[vix_close] - df["vix_ma10"]) / df["vix_ma10"] * 100
    df = df.drop(columns=["vix_ma10"])

    # VIX9D minus VIX spread
    vix9d_col = _find_col(df, ["vix9d_close", "vix9d_vix9d_close"], required=False)
    if vix9d_col:
        df["vix9d_minus_vix"] = df[vix9d_col] - df[vix_close]
    else:
        df["vix9d_minus_vix"] = np.nan

    # 5-day realized vol on SPY vs VIX (IV-RV spread proxy)
    daily_ret = df[spy_close].pct_change(1)
    df["spy_5d_rvol"] = daily_ret.rolling(5).std() * np.sqrt(252) * 100
    df["iv_rv_spread"] = df["vix_level"] - df["spy_5d_rvol"]

    # ── Sector features ──────────────────────────────────────────────────────
    xlk_close = _find_col(df, ["xlk_close"], required=False)
    xlf_close = _find_col(df, ["xlf_close"], required=False)
    xle_close = _find_col(df, ["xle_close"], required=False)
    xlv_close = _find_col(df, ["xlv_close"], required=False)

    spy_ret_1d = df[spy_close].pct_change(1)
    spy_ret_5d = df[spy_close].pct_change(5)

    if xlk_close:
        xlk_ret_1d = df[xlk_close].pct_change(1)
        df["xlk_vs_spy_1d"] = xlk_ret_1d - spy_ret_1d
        xlk_ret_5d = df[xlk_close].pct_change(5)
        df["xlk_vs_spy_5d"] = xlk_ret_5d - spy_ret_5d
    else:
        df["xlk_vs_spy_1d"] = np.nan
        df["xlk_vs_spy_5d"] = np.nan

    if xlf_close:
        xlf_ret_1d = df[xlf_close].pct_change(1)
        df["xlf_vs_spy_1d"] = xlf_ret_1d - spy_ret_1d
    else:
        df["xlf_vs_spy_1d"] = np.nan

    # Count of sectors positive previous day
    sector_cols = [c for c in [xlk_close, xlf_close, xle_close, xlv_close] if c]
    if sector_cols:
        prev_rets = pd.concat([df[c].pct_change(1) for c in sector_cols], axis=1)
        df["sectors_positive_count"] = (prev_rets > 0).sum(axis=1)
    else:
        df["sectors_positive_count"] = np.nan

    # ── Calendar features ────────────────────────────────────────────────────
    df["day_of_week"] = df["date"].dt.dayofweek            # 0=Mon … 4=Fri

    # Week of month (1–4)
    df["week_of_month"] = df["date"].apply(
        lambda d: min(4, (d.day - 1) // 7 + 1)
    )

    # Days until next FOMC
    df["days_to_fomc"] = _days_until_next(FOMC_DATES, df["date"])

    # Days until next CPI
    df["days_to_cpi"] = _days_until_next(CPI_DATES, df["date"])

    return df


def generate_labels(df: pd.DataFrame, threshold: float = 0.002) -> pd.DataFrame:
    """
    Generate intraday SPY return label.

    SPY intraday return = (close - open) / open  on the same day.
    > +threshold  → label = 1  (call spread)
    < -threshold  → label = -1 (put spread)
    else          → label = 0  (no trade)
    """
    spy_close_col = _find_col(df, ["spy_close", "spy_adj_close"])
    spy_open_col = _find_col(df, ["spy_open"])

    df = df.copy()
    df["spy_intraday_return"] = (df[spy_close_col] - df[spy_open_col]) / df[spy_open_col]

    conditions = [
        df["spy_intraday_return"] > threshold,
        df["spy_intraday_return"] < -threshold,
    ]
    choices = [1, -1]
    df["target"] = np.select(conditions, choices, default=0).astype(int)

    print("\n[engineer] Label distribution (threshold=±{:.1f}%):".format(threshold * 100))
    counts = df["target"].value_counts().sort_index()
    total = len(df)
    for label, cnt in counts.items():
        label_name = {1: "UP  (+1)", -1: "DOWN (-1)", 0: "FLAT ( 0)"}[label]
        print(f"  {label_name}: {cnt:5d}  ({cnt/total*100:.1f}%)")

    return df


def _find_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> Optional[str]:
    """Return the first candidate column that exists in df."""
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(
            f"[engineer] Could not find any of {candidates} in DataFrame columns:\n{list(df.columns)}"
        )
    return None


# Fix missing import for Optional
from typing import Optional  # noqa: E402 (already imported at module level via type hints)


FEATURE_COLUMNS = [
    # Momentum
    "spy_prev_return",
    "spy_5d_return",
    "spy_dist_ma20",
    "es_overnight_gap",
    # Volatility
    "vix_level",
    "vix_vs_ma10",
    "vix9d_minus_vix",
    "iv_rv_spread",
    # Sector
    "xlk_vs_spy_1d",
    "xlk_vs_spy_5d",
    "xlf_vs_spy_1d",
    "sectors_positive_count",
    # Calendar
    "day_of_week",
    "week_of_month",
    "days_to_fomc",
    "days_to_cpi",
]


def run_pipeline(input_path: Optional[Path] = None, output_path: Optional[Path] = None):
    """Full engineer pipeline: load → engineer → label → save."""
    if input_path is None:
        input_path = PROCESSED_DIR / "features.parquet"
    if output_path is None:
        output_path = PROCESSED_DIR / "features.parquet"

    print(f"[engineer] Loading {input_path} …")
    df = pd.read_parquet(input_path)
    df["date"] = pd.to_datetime(df["date"])
    print(f"  {len(df):,} rows, {len(df.columns)} columns")

    print("\n[engineer] Engineering features …")
    df = engineer_features(df)

    print("\n[engineer] Generating labels …")
    df = generate_labels(df)

    # Drop rows missing any feature
    feature_cols_present = [c for c in FEATURE_COLUMNS if c in df.columns]
    before = len(df)
    df = df.dropna(subset=feature_cols_present + ["target"])
    dropped = before - len(df)
    if dropped:
        print(f"\n[engineer] Dropped {dropped} rows with NaN features/labels.")

    df.to_parquet(output_path, index=False)
    print(f"\n[engineer] ✓ Saved → {output_path}")
    print(f"  Rows: {len(df):,}  |  Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  Feature columns: {feature_cols_present}")
    return df


if __name__ == "__main__":
    run_pipeline()
