"""
merge.py — Join all raw parquet files on trading date, output features.parquet.

No API calls. Reads only from data/raw/, writes to data/processed/.
"""

from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def _load_parquet(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"[merge] Required file not found: {path}\nRun the data pipeline first.")
    df = pd.read_parquet(path)
    print(f"[merge] Loaded {label}: {len(df):,} rows")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df


def main():
    print("[merge] Loading raw parquet files …")

    spy = _load_parquet(RAW_DIR / "spy_daily.parquet", "SPY/SPX")
    es = _load_parquet(RAW_DIR / "es_futures_daily.parquet", "ES futures")
    vix = _load_parquet(RAW_DIR / "vix_daily.parquet", "VIX")
    sectors = _load_parquet(RAW_DIR / "sectors_daily.parquet", "Sectors")

    # ── Standardise SPY columns ──────────────────────────────────────────────
    spy = spy.rename(columns={c: f"spy_{c}" for c in spy.columns
                               if c not in ("date", "symbol") and not c.startswith("spy_")})
    spy = spy.drop(columns=["symbol"], errors="ignore")

    # ── Standardise ES columns ───────────────────────────────────────────────
    es = es.rename(columns={c: f"es_{c}" for c in es.columns
                              if c not in ("date", "symbol", "es_overnight_gap")
                              and not c.startswith("es_")})
    es = es.drop(columns=["symbol"], errors="ignore")

    # ── Use SPY calendar as anchor ───────────────────────────────────────────
    print("\n[merge] Aligning to SPY trading calendar …")
    base = spy[["date"]].drop_duplicates().sort_values("date")

    # Merge all datasets
    df = base.merge(spy, on="date", how="left")
    df = df.merge(es, on="date", how="left")
    df = df.merge(vix, on="date", how="left")
    df = df.merge(sectors, on="date", how="left")

    initial_rows = len(df)
    print(f"[merge] After join: {initial_rows:,} rows")

    # ── Drop rows with missing values and log why ────────────────────────────
    print("\n[merge] Checking for missing values …")
    missing_report = df.isnull().sum()
    cols_with_missing = missing_report[missing_report > 0]
    if len(cols_with_missing):
        print("  Columns with NaNs:")
        for col, cnt in cols_with_missing.items():
            print(f"    {col}: {cnt} missing")
    else:
        print("  No missing values found.")

    # Drop rows where core SPY or VIX data is missing
    core_cols = [c for c in df.columns if c.startswith("spy_") or c.startswith("vix_")]
    before = len(df)
    df = df.dropna(subset=core_cols)
    dropped = before - len(df)
    if dropped:
        print(f"\n[merge] Dropped {dropped} rows with missing core SPY/VIX columns.")

    df = df.sort_values("date").reset_index(drop=True)

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = PROCESSED_DIR / "features.parquet"
    df.to_parquet(out_path, index=False)

    print(f"\n[merge] ✓ Saved merged data → {out_path}")
    print(f"  Rows : {len(df):,}")
    print(f"  Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  Columns ({len(df.columns)}): {list(df.columns)}")


if __name__ == "__main__":
    main()
