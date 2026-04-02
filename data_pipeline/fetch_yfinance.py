"""
fetch_yfinance.py — Fetch VIX, VIX9D, and sector ETFs via yfinance.

No Databento API calls. Date range is auto-detected from parsed dbn parquet files.
"""

import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV"]
VIX_TICKERS = ["^VIX", "^VIX9D"]


def _detect_date_range() -> tuple[str, str]:
    """
    Detect start/end dates from existing raw parquet files produced by parse_dbn.
    Falls back to a wide default if no files exist yet.
    """
    candidates = [
        RAW_DIR / "es_futures_daily.parquet",
        RAW_DIR / "spy_daily.parquet",
    ]

    all_dates = []
    for path in candidates:
        if path.exists():
            try:
                df = pd.read_parquet(path, columns=["date"])
                df["date"] = pd.to_datetime(df["date"])
                all_dates.extend(df["date"].dropna().tolist())
            except Exception:
                pass

    if all_dates:
        start = min(all_dates).strftime("%Y-%m-%d")
        end = pd.Timestamp.today().strftime("%Y-%m-%d")
        print(f"[fetch_yfinance] Auto-detected date range from dbn parquet: {start} → {end}")
    else:
        # Default fallback: 2018-01-01 to today
        start = "2018-01-01"
        end = pd.Timestamp.today().strftime("%Y-%m-%d")
        print(f"[fetch_yfinance] No dbn parquet found; using default range: {start} → {end}")

    return start, end


def _download(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download a single ticker via yfinance and return a clean DataFrame."""
    print(f"  Downloading {ticker} …", end=" ", flush=True)
    raw = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )
    if raw.empty:
        print("EMPTY")
        return pd.DataFrame()

    # Flatten multi-level columns if present
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0].lower() for c in raw.columns]
    else:
        raw.columns = [c.lower() for c in raw.columns]

    raw = raw.reset_index()
    raw = raw.rename(columns={"Date": "date", "index": "date"})
    raw["date"] = pd.to_datetime(raw["date"]).dt.normalize()
    print(f"{len(raw)} rows  ({raw['date'].min().date()} → {raw['date'].max().date()})")
    return raw


def fetch_vix(start: str, end: str):
    """Fetch ^VIX and ^VIX9D, merge on date, save to vix_daily.parquet."""
    vix_df = _download("^VIX", start, end)
    vix9d_df = _download("^VIX9D", start, end)

    if vix_df.empty:
        print("[fetch_yfinance] WARNING: ^VIX download failed.")
        return

    # Rename columns to avoid clash on merge
    vix_cols = {c: f"vix_{c}" for c in vix_df.columns if c != "date"}
    vix_df = vix_df.rename(columns=vix_cols)

    if not vix9d_df.empty:
        vix9d_cols = {c: f"vix9d_{c}" for c in vix9d_df.columns if c != "date"}
        vix9d_df = vix9d_df.rename(columns=vix9d_cols)
        merged = vix_df.merge(vix9d_df, on="date", how="left")
    else:
        print("[fetch_yfinance] WARNING: ^VIX9D download failed; VIX9D columns will be NaN.")
        merged = vix_df

    merged = merged.sort_values("date").drop_duplicates(subset=["date"])
    out_path = RAW_DIR / "vix_daily.parquet"
    merged.to_parquet(out_path, index=False)
    print(f"[fetch_yfinance] Saved VIX → {out_path} ({len(merged):,} rows)")


def fetch_sectors(start: str, end: str):
    """Fetch sector ETFs and save to sectors_daily.parquet."""
    frames = []
    for ticker in SECTOR_ETFS:
        df = _download(ticker, start, end)
        if df.empty:
            print(f"[fetch_yfinance] WARNING: {ticker} download returned empty DataFrame.")
            continue
        df = df.rename(columns={c: f"{ticker.lower()}_{c}" for c in df.columns if c != "date"})
        frames.append(df)

    if not frames:
        print("[fetch_yfinance] ERROR: No sector ETF data retrieved.")
        return

    merged = frames[0]
    for df in frames[1:]:
        merged = merged.merge(df, on="date", how="outer")

    merged = merged.sort_values("date").drop_duplicates(subset=["date"])
    out_path = RAW_DIR / "sectors_daily.parquet"
    merged.to_parquet(out_path, index=False)
    print(f"[fetch_yfinance] Saved sectors → {out_path} ({len(merged):,} rows)")


def main():
    print("[fetch_yfinance] Starting …")
    start, end = _detect_date_range()

    print("\n[fetch_yfinance] Fetching VIX data:")
    fetch_vix(start, end)

    print("\n[fetch_yfinance] Fetching sector ETF data:")
    fetch_sectors(start, end)

    print("\n[fetch_yfinance] Done.")


if __name__ == "__main__":
    main()
