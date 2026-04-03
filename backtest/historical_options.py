"""
backtest/historical_options.py — Fetch and cache real historical SPX options pricing.

Pricing strategy (in order of preference):
  1. Massive API snapshot with as_of date — same endpoint family used in live
     trading; fetches the snapshot as it appeared at market open on trade_date.
     15-min delay is inherent: we request 9:30am data which is what the live
     delayed feed shows at 9:45am entry time.
  2. Black-Scholes with VIX — uses the VIX level already in the features
     DataFrame as the annualized IV proxy.  Mathematically sound for 0DTE SPX.
  3. 35% approximation — only if both above fail (no API key, no VIX data).

Results are cached to data/cache/options_prices.parquet so the API is only
called once per (date, contract_type, spread_width) combination.
"""

import math
import os
import time
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import requests
import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

CACHE_DIR = PROJECT_ROOT / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PATH = CACHE_DIR / "options_prices.parquet"

MASSIVE_API_KEY = os.environ.get("MASSIVE_API_KEY")
MASSIVE_BASE_URL = "https://api.massive.com"
SPX_MULTIPLIER = 100

# Risk-free rate used for Black-Scholes (approximate; no large sensitivity for 0DTE)
_BS_RISK_FREE = 0.05

# 0DTE at 9:30am ET: 6.5 hours until 4pm close → fraction of calendar year
_T_0DTE_930AM = 6.5 / 8_760


# ── API helpers ────────────────────────────────────────────────────────────────

def _headers() -> dict:
    if not MASSIVE_API_KEY:
        raise EnvironmentError("[historical_options] MASSIVE_API_KEY not set")
    return {"Authorization": f"Bearer {MASSIVE_API_KEY}"}


def _get(path: str, params: dict = None, retries: int = 3) -> dict:
    url = f"{MASSIVE_BASE_URL}{path}"
    for attempt in range(retries):
        try:
            resp = requests.get(
                url, headers=_headers(), params=params or {}, timeout=15
            )
            if resp.status_code == 429:
                wait = 2 ** attempt
                print(f"[historical_options]   Rate limited, waiting {wait}s …")
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                raise RuntimeError(
                    f"[historical_options] API {resp.status_code} on {path}: "
                    f"{resp.text[:200]}"
                )
            return resp.json()
        except requests.RequestException as exc:
            if attempt == retries - 1:
                raise
            time.sleep(1 + attempt)
    raise RuntimeError(f"[historical_options] All retries exhausted for {path}")


# ── OCC ticker ─────────────────────────────────────────────────────────────────

def _occ_ticker(underlying: str, expiry: date, contract_type: str, strike: float) -> str:
    """Build OCC-format options ticker, e.g. O:SPX230103C03810000."""
    exp_str = expiry.strftime("%y%m%d")
    c_type = "C" if contract_type.lower() == "call" else "P"
    strike_int = int(round(strike * 1000))
    return f"O:{underlying}{exp_str}{c_type}{str(strike_int).zfill(8)}"


# ── Strike selection ───────────────────────────────────────────────────────────

def _atm_strikes(
    contract_type: str, spx_price: float, spread_width: int
) -> tuple[float, float]:
    """
    Round spx_price to nearest 5-point increment for first OTM strike.
    Mirrors dashboard/massive.py for consistency.
    """
    if contract_type == "call":
        long_strike = round(spx_price / 5) * 5
        if long_strike <= spx_price:
            long_strike += 5
    else:
        long_strike = round(spx_price / 5) * 5
        if long_strike >= spx_price:
            long_strike -= 5
    short_strike = (
        long_strike + spread_width
        if contract_type == "call"
        else long_strike - spread_width
    )
    return float(long_strike), float(short_strike)


# ── Black-Scholes fallback ─────────────────────────────────────────────────────

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _bs_option_price(
    contract_type: str, S: float, K: float, T: float, r: float, sigma: float
) -> float:
    """European option price via Black-Scholes."""
    if T <= 0:
        return max(S - K, 0.0) if contract_type == "call" else max(K - S, 0.0)
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    if contract_type == "call":
        return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    else:
        return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def _bs_spread_pricing(
    contract_type: str,
    spx_price: float,
    long_strike: float,
    short_strike: float,
    vix: float,
) -> Optional[dict]:
    """
    Black-Scholes spread pricing for 0DTE at 9:30am ET.
    VIX is used as the annualized IV proxy (standard market practice).
    T = 6.5 hrs (9:30am → 4pm close) as fraction of calendar year.
    """
    sigma = vix / 100.0
    if sigma <= 0 or spx_price <= 0:
        return None

    long_price = _bs_option_price(
        contract_type, spx_price, long_strike, _T_0DTE_930AM, _BS_RISK_FREE, sigma
    )
    short_price = _bs_option_price(
        contract_type, spx_price, short_strike, _T_0DTE_930AM, _BS_RISK_FREE, sigma
    )

    debit_per_share = long_price - short_price
    if debit_per_share <= 0:
        return None

    debit_dollars = debit_per_share * SPX_MULTIPLIER
    spread_width = abs(long_strike - short_strike)
    max_value_dollars = spread_width * SPX_MULTIPLIER
    max_gain_dollars = max_value_dollars - debit_dollars

    if max_gain_dollars <= 0:
        return None

    return {
        "debit_dollars": round(debit_dollars, 2),
        "max_gain_dollars": round(max_gain_dollars, 2),
        "debit_per_share": round(debit_per_share, 4),
        "long_strike": long_strike,
        "short_strike": short_strike,
        "source": "black_scholes_vix",
    }


# ── Massive API snapshot (historical) ─────────────────────────────────────────

def _fetch_snapshot_for_contract(
    underlying: str, occ_ticker: str, trade_date: date
) -> Optional[dict]:
    """
    Fetch a single-contract snapshot with as_of date.
    Uses the same /v3/snapshot/options endpoint as live trading.
    The as_of date gives us the snapshot as it appeared on that trading day,
    which with the live feed's 15-min delay corresponds to the 9:45am view.
    """
    try:
        data = _get(
            f"/v3/snapshot/options/{underlying}/{occ_ticker}",
            {"as_of": trade_date.isoformat()},
        )
        return data.get("results", {}) or None
    except Exception as exc:
        # Only print non-SSL errors as warnings; SSL means endpoint not supported
        msg = str(exc)
        if "SSL" not in msg and "ssl" not in msg:
            print(f"[historical_options]   Snapshot fetch failed for {occ_ticker}: {exc}")
        return None


def _extract_bid_ask(snapshot: dict) -> tuple[Optional[float], Optional[float]]:
    """Extract bid and ask from a snapshot result."""
    quote = snapshot.get("last_quote", {})
    bid = quote.get("bid")
    ask = quote.get("ask")
    mid = quote.get("midpoint")
    # Use midpoint as fallback for either side
    return (bid or mid), (ask or mid)


# ── Main pricing function ──────────────────────────────────────────────────────

def fetch_spread_pricing(
    trade_date: date,
    contract_type: str,
    spx_price: float,
    spread_width: int,
    vix: Optional[float] = None,
) -> Optional[dict]:
    """
    Fetch real spread pricing for a 0DTE trade on trade_date.

    Order of attempts:
      1. Massive API snapshot with as_of=trade_date (real market data, 15-min delay)
      2. Black-Scholes with VIX (if vix is provided)
      3. Returns None → caller uses 35% approximation

    15-min delay note: the snapshot as_of endpoint returns data as it appeared
    at market open, matching what the live 15-min delayed feed shows at 9:45am.
    """
    long_strike, short_strike = _atm_strikes(contract_type, spx_price, spread_width)

    # ── Attempt 1: API snapshot with as_of ─────────────────────────────────────
    if MASSIVE_API_KEY:
        long_ticker = _occ_ticker("SPX", trade_date, contract_type, long_strike)
        short_ticker = _occ_ticker("SPX", trade_date, contract_type, short_strike)

        long_snap = _fetch_snapshot_for_contract("SPX", long_ticker, trade_date)
        short_snap = _fetch_snapshot_for_contract("SPX", short_ticker, trade_date)

        if long_snap and short_snap:
            _, long_ask = _extract_bid_ask(long_snap)
            short_bid, _ = _extract_bid_ask(short_snap)

            if long_ask is not None and short_bid is not None:
                debit_per_share = float(long_ask) - float(short_bid)
                if debit_per_share > 0:
                    debit_dollars = debit_per_share * SPX_MULTIPLIER
                    max_gain_dollars = spread_width * SPX_MULTIPLIER - debit_dollars
                    if max_gain_dollars > 0:
                        return {
                            "debit_dollars": round(debit_dollars, 2),
                            "max_gain_dollars": round(max_gain_dollars, 2),
                            "debit_per_share": round(debit_per_share, 4),
                            "long_strike": long_strike,
                            "short_strike": short_strike,
                            "source": "api_snapshot",
                        }

    # ── Attempt 2: Black-Scholes with VIX ──────────────────────────────────────
    if vix is not None and vix > 0:
        result = _bs_spread_pricing(
            contract_type, spx_price, long_strike, short_strike, vix
        )
        if result is not None:
            return result

    return None  # caller falls back to 35%


# ── Cache build ────────────────────────────────────────────────────────────────

def _build_cache(
    df: pd.DataFrame,
    spread_widths: list[int],
    api_delay_seconds: float = 0.2,
) -> pd.DataFrame:
    """
    Fetch pricing for every (date, contract_type, spread_width) in df.
    Uses spy_close × 10 as SPX price approximation and vix_level for BS fallback.
    """
    spy_col = next(
        (c for c in df.columns if c in ("spy_close", "spy_adj_close")), None
    )
    vix_col = next(
        (c for c in df.columns if c in ("vix_level", "vix_close", "vix_vix_close")), None
    )
    if spy_col is None:
        print("[historical_options] No spy_close column — cannot build options cache")
        return pd.DataFrame()

    dates = sorted(df["date"].dt.date.unique())
    total = len(dates) * len(spread_widths) * 2
    done = 0
    api_hits = bs_hits = fallback_hits = 0
    rows = []

    print(
        f"[historical_options] Building options cache: "
        f"{len(dates)} dates × {len(spread_widths)} widths × 2 directions "
        f"= {total} slots"
    )
    if not MASSIVE_API_KEY:
        print("[historical_options] No MASSIVE_API_KEY — using Black-Scholes only")

    for trade_date in dates:
        date_row = df[df["date"].dt.date == trade_date].iloc[0]
        spx_price = float(date_row[spy_col]) * 10
        vix = float(date_row[vix_col]) if vix_col and pd.notna(date_row[vix_col]) else None

        for spread_width in spread_widths:
            for contract_type in ("call", "put"):
                done += 1
                if done % 50 == 0 or done == total:
                    print(
                        f"[historical_options]   {done}/{total}  "
                        f"(api={api_hits} bs={bs_hits} fallback={fallback_hits})"
                    )

                result = fetch_spread_pricing(
                    trade_date, contract_type, spx_price, spread_width, vix
                )

                if result is not None:
                    if result["source"] == "api_snapshot":
                        api_hits += 1
                    else:
                        bs_hits += 1
                    rows.append(
                        {
                            "date": str(trade_date),
                            "contract_type": contract_type,
                            "spread_width": spread_width,
                            "debit_dollars": result["debit_dollars"],
                            "max_gain_dollars": result["max_gain_dollars"],
                            "debit_per_share": result["debit_per_share"],
                            "long_strike": result["long_strike"],
                            "short_strike": result["short_strike"],
                            "source": result["source"],
                        }
                    )
                else:
                    fallback_hits += 1

                if MASSIVE_API_KEY:
                    time.sleep(api_delay_seconds)

    print(
        f"[historical_options] Done: api_snapshot={api_hits}  "
        f"black_scholes={bs_hits}  35pct_fallback={fallback_hits}"
    )
    return pd.DataFrame(rows)


# ── Public interface ───────────────────────────────────────────────────────────

def load_options_cache(
    df: pd.DataFrame,
    spread_widths: list[int],
    force_refresh: bool = False,
) -> dict:
    """
    Load the on-disk options pricing cache, fetching any missing dates.

    Returns lookup dict keyed by (date_str, contract_type, spread_width):
        {("2023-01-10", "call", 3): {"debit_dollars": 112.5, "max_gain_dollars": 187.5}, ...}

    Trades whose key is absent fall back to the 35% approximation in engine.py.
    """
    cache_df = pd.DataFrame()

    if CACHE_PATH.exists() and not force_refresh:
        cache_df = pd.read_parquet(CACHE_PATH)
        print(
            f"[historical_options] Loaded cache: {len(cache_df)} rows "
            f"from {CACHE_PATH}"
        )

    # Determine which (date, width, direction) combos are missing
    needed_dates = {str(d) for d in df["date"].dt.date.unique()}
    if not cache_df.empty:
        cached_keys = set(
            zip(
                cache_df["date"].astype(str),
                cache_df["contract_type"],
                cache_df["spread_width"].astype(int),
            )
        )
        missing_dates = {
            d for d in needed_dates
            for w in spread_widths
            for ct in ("call", "put")
            if (d, ct, w) not in cached_keys
        }
    else:
        missing_dates = needed_dates

    if missing_dates:
        print(f"[historical_options] Fetching {len(missing_dates)} missing dates …")
        missing_df = df[df["date"].dt.date.apply(str).isin(missing_dates)]
        new_data = _build_cache(missing_df, spread_widths)
        if not new_data.empty:
            cache_df = pd.concat([cache_df, new_data], ignore_index=True)
            cache_df = cache_df.drop_duplicates(
                subset=["date", "contract_type", "spread_width"]
            )
            cache_df.to_parquet(CACHE_PATH, index=False)
            print(
                f"[historical_options] Cache saved → {CACHE_PATH} "
                f"({len(cache_df)} rows)"
            )

    # Build O(1) lookup dict
    lookup: dict = {}
    for _, row in cache_df.iterrows():
        key = (str(row["date"]), str(row["contract_type"]), int(row["spread_width"]))
        lookup[key] = {
            "debit_dollars": float(row["debit_dollars"]),
            "max_gain_dollars": float(row["max_gain_dollars"]),
        }

    total_slots = len(needed_dates) * len(spread_widths) * 2
    covered = len(lookup)
    print(
        f"[historical_options] Cache ready: {covered}/{total_slots} slots "
        f"({100*covered/max(total_slots,1):.0f}% covered — remainder uses 35% fallback)"
    )
    return lookup
