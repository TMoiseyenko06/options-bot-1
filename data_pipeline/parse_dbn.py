"""
parse_dbn.py — Read local .dbn files, route to parquet by symbol.

No API calls anywhere in this file.
All data read from data/dbn/ directory.
"""

import os
import sys
from pathlib import Path
from typing import Optional

import databento as db
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DBN_DIR = PROJECT_ROOT / "data" / "dbn"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

RAW_DIR.mkdir(parents=True, exist_ok=True)

# Symbol routing keys
ES_KEYWORDS = {"ES", "MES"}
SPY_KEYWORDS = {"SPY", "SPX"}


def _detect_symbol(store: db.DBNStore) -> Optional[str]:
    """Extract the primary symbol from DBNStore metadata."""
    metadata = store.metadata
    symbols = getattr(metadata, "symbols", None) or []
    if symbols:
        return symbols[0].upper()
    # Fallback: try instrument_id map
    try:
        sym_map = store.symbology_map
        if sym_map:
            return next(iter(sym_map.values())).upper()
    except Exception:
        pass
    return None


def _is_es(symbol: str) -> bool:
    return any(kw in symbol.upper() for kw in ES_KEYWORDS)


def _is_spy(symbol: str) -> bool:
    return any(kw in symbol.upper() for kw in SPY_KEYWORDS)


def _extract_overnight_gap(df: pd.DataFrame, schema: str) -> Optional[pd.Series]:
    """
    Compute overnight gap from ES intraday data when available.

    For minute/hourly data: prior 4pm ET close → next 9:30am ET open.
    For daily data: (open - prior_close) / prior_close.
    Returns a Series indexed by date with name 'es_overnight_gap'.
    """
    df = df.copy()

    # Normalise timestamp column
    ts_col = None
    for c in ("ts_event", "ts_recv", "timestamp"):
        if c in df.columns:
            ts_col = c
            break
    if ts_col is None:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "ts_event"})
            ts_col = "ts_event"
        else:
            return None

    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
    df = df.sort_values(ts_col)

    schema_lower = schema.lower() if schema else ""
    is_intraday = any(s in schema_lower for s in ("mbp", "ohlcv-1m", "ohlcv-1h", "trades", "mbo"))

    if is_intraday:
        # Convert to US/Eastern
        df["ts_et"] = df[ts_col].dt.tz_convert("US/Eastern")
        df["date"] = df["ts_et"].dt.date
        df["hour"] = df["ts_et"].dt.hour
        df["minute"] = df["ts_et"].dt.minute

        # Prior 4 pm close: last bar at 16:00 ET
        close_mask = (df["hour"] == 16) & (df["minute"] == 0)
        prior_close = (
            df[close_mask]
            .groupby("date")["close"]
            .last()
            .rename("prior_close")
        )

        # Next 9:30 am open
        open_mask = (df["hour"] == 9) & (df["minute"] == 30)
        next_open = (
            df[open_mask]
            .groupby("date")["open"]
            .first()
            .rename("next_open")
        )

        prior_close.index = pd.to_datetime(prior_close.index)
        next_open.index = pd.to_datetime(next_open.index)

        # Align: open on date T corresponds to close on date T-1
        combined = pd.DataFrame({"next_open": next_open})
        combined["prior_close"] = prior_close.shift(1)
        combined = combined.dropna()
        gap = (combined["next_open"] - combined["prior_close"]) / combined["prior_close"]
        gap.name = "es_overnight_gap"
        return gap

    else:
        # Daily fallback
        if "open" not in df.columns or "close" not in df.columns:
            return None
        df["date"] = pd.to_datetime(df[ts_col]).dt.normalize()
        daily = df.groupby("date").agg({"open": "first", "close": "last"}).sort_index()
        gap = (daily["open"] - daily["close"].shift(1)) / daily["close"].shift(1)
        gap.name = "es_overnight_gap"
        return gap.dropna()


def _to_daily_ohlcv(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    """Aggregate any resolution DataFrame to daily OHLCV."""
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
    df["date"] = df[ts_col].dt.normalize().dt.tz_localize(None)
    agg_cols = {c: "first" for c in ("open",) if c in df.columns}
    agg_cols.update({c: "max" for c in ("high",) if c in df.columns})
    agg_cols.update({c: "min" for c in ("low",) if c in df.columns})
    agg_cols.update({c: "last" for c in ("close",) if c in df.columns})
    agg_cols.update({c: "sum" for c in ("volume",) if c in df.columns})
    if not agg_cols:
        return df
    daily = df.groupby("date").agg(agg_cols)
    daily.index.name = "date"
    return daily.reset_index()


def _extract_intraday_market_hours(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    """
    Extract regular market hours bars (9:30am–4:00pm ET) from a minute-resolution DataFrame.
    Returns columns: ts_et, date, open, high, low, close, volume.
    """
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
    df["ts_et"] = df[ts_col].dt.tz_convert("US/Eastern").dt.tz_localize(None)
    df["date"] = df["ts_et"].dt.normalize()
    df["hour"] = df["ts_et"].dt.hour
    df["minute"] = df["ts_et"].dt.minute

    # Regular market hours: 9:30am to 3:59pm ET inclusive
    market_mask = (
        ((df["hour"] == 9) & (df["minute"] >= 30)) |
        ((df["hour"] >= 10) & (df["hour"] <= 15)) |
        ((df["hour"] == 15) & (df["minute"] <= 59))
    )
    # Simpler: keep 9:30 through 15:59
    market_mask = (
        (df["ts_et"].dt.hour * 60 + df["ts_et"].dt.minute >= 9 * 60 + 30) &
        (df["ts_et"].dt.hour * 60 + df["ts_et"].dt.minute <= 15 * 60 + 59)
    )

    df = df[market_mask].copy()

    keep_cols = ["ts_et", "date"]
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            keep_cols.append(col)

    return df[keep_cols].reset_index(drop=True)


def parse_all_dbn_files():
    """Main entry point: scan, parse, route, and save parquet files."""
    dbn_files = list(DBN_DIR.glob("*.dbn")) + list(DBN_DIR.glob("*.dbn.zst"))

    if not dbn_files:
        print(f"[parse_dbn] No .dbn files found in {DBN_DIR}")
        print("[parse_dbn] Place .dbn files in data/dbn/ and re-run.")
        return

    print(f"[parse_dbn] Found {len(dbn_files)} .dbn file(s):")
    for f in dbn_files:
        print(f"  {f.name}")

    es_frames = []
    spy_frames = []
    spy_intraday_frames = []   # 1-minute SPY bars for backtest exit simulation
    es_gap_series = []

    for fpath in dbn_files:
        print(f"\n[parse_dbn] Opening {fpath.name} …")
        try:
            store = db.DBNStore.from_file(str(fpath))
        except Exception as e:
            print(f"  ERROR opening file: {e}")
            continue

        metadata = store.metadata
        schema = str(getattr(metadata, "schema", "unknown"))
        symbol = _detect_symbol(store)
        dataset = str(getattr(metadata, "dataset", ""))

        print(f"  schema  : {schema}")
        print(f"  symbol  : {symbol}")
        print(f"  dataset : {dataset}")

        try:
            df = store.to_df()
        except Exception as e:
            print(f"  ERROR converting to DataFrame: {e}")
            continue

        if df.empty:
            print("  WARNING: DataFrame is empty, skipping.")
            continue

        # Identify timestamp column
        ts_col = None
        for c in ("ts_event", "ts_recv", "timestamp"):
            if c in df.columns:
                ts_col = c
                break
        if ts_col is None and isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "ts_event"})
            ts_col = "ts_event"

        if ts_col:
            df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
            date_min = df[ts_col].min()
            date_max = df[ts_col].max()
        else:
            date_min = date_max = "unknown"

        print(f"  rows    : {len(df):,}")
        print(f"  dates   : {date_min} → {date_max}")
        print(f"  columns : {list(df.columns)}")

        if symbol is None:
            print("  WARNING: Could not detect symbol, skipping.")
            continue

        schema_lower = schema.lower()
        is_minute = "ohlcv-1m" in schema_lower or "ohlcv-1s" in schema_lower
        is_intraday = is_minute or any(
            s in schema_lower for s in ("mbp", "ohlcv-1h", "trades", "mbo")
        )

        # Route by symbol
        if _is_es(symbol):
            print(f"  → Routing as ES futures")
            # Compute overnight gap before aggregating
            gap = _extract_overnight_gap(df, schema)
            if gap is not None:
                es_gap_series.append(gap)
                print(f"  overnight gap rows: {len(gap)}")

            if ts_col:
                daily = _to_daily_ohlcv(df, ts_col)
                daily["symbol"] = symbol
                es_frames.append(daily)

        elif _is_spy(symbol):
            print(f"  → Routing as SPY/SPX equity")
            if ts_col:
                daily = _to_daily_ohlcv(df, ts_col)
                daily["symbol"] = symbol
                spy_frames.append(daily)

            # If 1-minute data: also save raw intraday bars for backtest
            if is_minute and ts_col and "open" in df.columns:
                intraday = _extract_intraday_market_hours(df, ts_col)
                if not intraday.empty:
                    spy_intraday_frames.append(intraday)
                    print(f"  → Also saving {len(intraday):,} intraday 1-min bars for backtest")
        else:
            print(f"  → Symbol '{symbol}' not matched to ES or SPY, skipping.")

    # --- Save ES futures ---
    if es_frames:
        es_df = pd.concat(es_frames, ignore_index=True)
        if "date" in es_df.columns:
            es_df["date"] = pd.to_datetime(es_df["date"])
            es_df = es_df.drop_duplicates(subset=["date"]).sort_values("date")

        # Attach overnight gap
        if es_gap_series:
            all_gaps = pd.concat(es_gap_series).sort_index()
            all_gaps = all_gaps[~all_gaps.index.duplicated(keep="last")]
            gap_df = all_gaps.reset_index().rename(columns={"index": "date", 0: "es_overnight_gap"})
            gap_df.columns = ["date", "es_overnight_gap"]
            gap_df["date"] = pd.to_datetime(gap_df["date"])
            es_df = es_df.merge(gap_df, on="date", how="left")

        out_path = RAW_DIR / "es_futures_daily.parquet"
        es_df.to_parquet(out_path, index=False)
        print(f"\n[parse_dbn] Saved ES futures → {out_path} ({len(es_df):,} rows)")
        print(f"  date range: {es_df['date'].min()} → {es_df['date'].max()}")

    # --- Save SPY/SPX daily ---
    if spy_frames:
        spy_df = pd.concat(spy_frames, ignore_index=True)
        if "date" in spy_df.columns:
            spy_df["date"] = pd.to_datetime(spy_df["date"])
            spy_df = spy_df.drop_duplicates(subset=["date"]).sort_values("date")

        out_path = RAW_DIR / "spy_daily.parquet"
        spy_df.to_parquet(out_path, index=False)
        print(f"\n[parse_dbn] Saved SPY/SPX daily → {out_path} ({len(spy_df):,} rows)")
        print(f"  date range: {spy_df['date'].min()} → {spy_df['date'].max()}")

    # --- Save SPY 1-minute intraday (market hours only) ---
    if spy_intraday_frames:
        intraday_df = pd.concat(spy_intraday_frames, ignore_index=True)
        intraday_df["ts_et"] = pd.to_datetime(intraday_df["ts_et"])
        intraday_df["date"] = pd.to_datetime(intraday_df["date"])
        # Deduplicate on exact timestamp
        intraday_df = intraday_df.drop_duplicates(subset=["ts_et"]).sort_values("ts_et")

        out_path = RAW_DIR / "spy_intraday_1m.parquet"
        intraday_df.to_parquet(out_path, index=False)
        print(f"\n[parse_dbn] Saved SPY 1-min intraday → {out_path} ({len(intraday_df):,} bars)")
        print(f"  date range: {intraday_df['date'].min()} → {intraday_df['date'].max()}")
    else:
        print("\n[parse_dbn] No 1-minute SPY data found — backtest will use daily OHLCV fallback.")

    if not es_frames and not spy_frames:
        print("\n[parse_dbn] No recognised symbols processed. Check your .dbn files.")

    print("\n[parse_dbn] Done.")


if __name__ == "__main__":
    parse_all_dbn_files()
