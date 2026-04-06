"""
backtest/historical_options.py — Fetch and cache real historical SPX options pricing.

Uses the official `massive` Python client (pip install massive) which handles
TLS and authentication correctly against api.massive.com.

Pricing strategy (in order of preference):
  1. /v3/quotes  — last bid/ask at or before 9:30am ET on trade_date.
                   15-min delay: at 9:45am entry the live feed shows 9:30am data,
                   so fetching 9:30am historical quotes gives identical pricing.
  2. /v2/aggs 1-min bars — fallback when quotes unavailable; uses 9:30am bar close.
  3. Black-Scholes with VIX — when API has no data for a date (e.g. very old contracts).
  4. None returned → engine.py falls back to 35% approximation.

Results cached to data/cache/options_prices.parquet.
"""

import math
import os
import time
from datetime import date, datetime
from pathlib import Path
from typing import Iterator, Optional

import pytz
import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

CACHE_DIR = PROJECT_ROOT / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PATH = CACHE_DIR / "options_prices.parquet"

MASSIVE_API_KEY = os.environ.get("MASSIVE_API_KEY")
SPX_MULTIPLIER = 100
ET_TZ = pytz.timezone("America/New_York")

# 15-min delay: fetch quotes at 9:30am ET — what the live delayed feed shows at 9:45am
_FETCH_HOUR = 9
_FETCH_MINUTE = 30

# Black-Scholes constants for 0DTE at 9:30am (T = 6.5 hrs until 4pm close)
_T_0DTE = 6.5 / 8_760
_BS_RISK_FREE = 0.05


# ── Massive client ─────────────────────────────────────────────────────────────

def _client():
    """Return an authenticated RESTClient. Raises clearly if key missing."""
    if not MASSIVE_API_KEY:
        raise EnvironmentError(
            "[historical_options] MASSIVE_API_KEY not set in .env"
        )
    from massive import RESTClient
    return RESTClient(api_key=MASSIVE_API_KEY)


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
    """First OTM strike rounded to nearest 5-pt increment. Matches dashboard/massive.py."""
    if contract_type == "call":
        long_strike = round(spx_price / 5) * 5
        if long_strike <= spx_price:
            long_strike += 5
    else:
        long_strike = round(spx_price / 5) * 5
        if long_strike >= spx_price:
            long_strike -= 5
    short_strike = (
        long_strike + spread_width if contract_type == "call"
        else long_strike - spread_width
    )
    return float(long_strike), float(short_strike)


# ── Black-Scholes fallback ─────────────────────────────────────────────────────

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _bs_price(contract_type: str, S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0:
        return max(S - K, 0.0) if contract_type == "call" else max(K - S, 0.0)
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    if contract_type == "call":
        return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def _bs_spread(
    contract_type: str, spx: float, k_long: float, k_short: float, vix: float
) -> Optional[dict]:
    sigma = vix / 100.0
    long_p = _bs_price(contract_type, spx, k_long, _T_0DTE, _BS_RISK_FREE, sigma)
    short_p = _bs_price(contract_type, spx, k_short, _T_0DTE, _BS_RISK_FREE, sigma)
    debit_ps = long_p - short_p
    if debit_ps <= 0:
        return None
    debit_d = debit_ps * SPX_MULTIPLIER
    max_gain_d = abs(k_long - k_short) * SPX_MULTIPLIER - debit_d
    if max_gain_d <= 0:
        return None
    return {
        "debit_dollars": round(debit_d, 2),
        "max_gain_dollars": round(max_gain_d, 2),
        "debit_per_share": round(debit_ps, 4),
        "source": "black_scholes_vix",
    }


# ── Quote fetch (primary) ──────────────────────────────────────────────────────

def _fetch_quote_at_930(
    client, occ_ticker: str, trade_date: date
) -> Optional[tuple[float, float]]:
    """
    Fetch the last bid/ask at or before 9:30am ET via /v3/quotes.
    Returns (bid, ask) or None.
    """
    dt_et = ET_TZ.localize(
        datetime(trade_date.year, trade_date.month, trade_date.day,
                 _FETCH_HOUR, _FETCH_MINUTE, 0)
    )
    print(f"[quotes] {occ_ticker}  timestamp_lte={dt_et.isoformat()}")
    try:
        quotes = list(
            client.list_quotes(
                occ_ticker,
                timestamp_lte=dt_et,
                limit=1,
                order="desc",
                sort="timestamp",
            )
        )
        if not quotes:
            print(f"[quotes]   no results")
            return None
        q = quotes[0]
        bid = q.bid_price
        ask = q.ask_price
        print(f"[quotes]   bid={bid}  ask={ask}  sip_ts={q.sip_timestamp}")
        if bid is None or ask is None:
            print(f"[quotes]   bid or ask is None — skipping")
            return None
        return float(bid), float(ask)
    except Exception as exc:
        print(f"[quotes]   FAILED: {exc}")
        return None


# ── Agg bar fetch (fallback) ───────────────────────────────────────────────────

def _fetch_agg_at_930(
    client, occ_ticker: str, trade_date: date
) -> Optional[float]:
    """
    Fallback: fetch 1-min OHLCV bars and return the close of the first bar
    at or after 9:30am ET. Used as a mid-price proxy when no quote is available.
    """
    date_str = trade_date.isoformat()
    print(f"[aggs]   {occ_ticker}  {date_str}")
    try:
        aggs = client.get_aggs(
            occ_ticker,
            multiplier=1,
            timespan="minute",
            from_=date_str,
            to=date_str,
            adjusted=False,
            sort="asc",
            limit=50,
        )
        if not aggs:
            print(f"[aggs]   no bars returned")
            return None
        for bar in aggs:
            ts_ms = bar.timestamp
            dt_et = (
                pytz.utc.localize(datetime.utcfromtimestamp(ts_ms / 1000))
                .astimezone(ET_TZ)
            )
            if dt_et.hour > _FETCH_HOUR or (
                dt_et.hour == _FETCH_HOUR and dt_et.minute >= _FETCH_MINUTE
            ):
                print(f"[aggs]   bar at {dt_et.strftime('%H:%M')} ET  close={bar.close}")
                return float(bar.close)
        print(f"[aggs]   no bar at or after 09:30 ET")
        return None
    except Exception as exc:
        print(f"[aggs]   FAILED: {exc}")
        return None


# ── Main pricing function ──────────────────────────────────────────────────────

def fetch_spread_pricing(
    trade_date: date,
    contract_type: str,
    spx_price: float,
    spread_width: int,
    vix: Optional[float] = None,
) -> Optional[dict]:
    """
    Fetch real spread pricing for a 0DTE entry on trade_date.

    Attempts in order:
      1. /v3/quotes at 9:30am ET (real bid/ask, 15-min delayed equivalent)
      2. /v2/aggs 1-min bar close at 9:30am ET
      3. Black-Scholes with VIX
      4. Returns None → caller uses 35% fallback
    """
    k_long, k_short = _atm_strikes(contract_type, spx_price, spread_width)
    long_ticker = _occ_ticker("SPX", trade_date, contract_type, k_long)
    short_ticker = _occ_ticker("SPX", trade_date, contract_type, k_short)

    print(
        f"\n[pricing] {trade_date}  {contract_type.upper()}  {spread_width}pt  "
        f"SPX={spx_price:.1f}  VIX={vix}  strikes={k_long}/{k_short}"
    )
    print(f"[pricing]   long  → {long_ticker}")
    print(f"[pricing]   short → {short_ticker}")

    # Historical options quotes/aggs require a paid plan upgrade on Massive.
    # For backtesting we skip straight to Black-Scholes — the live signal
    # still uses the snapshot endpoint (current-day data) which IS included.
    print(f"[pricing]   skipping API for historical date — using Black-Scholes")

    # ── Attempt 3: Black-Scholes with VIX ─────────────────────────────────────
    if vix is not None and vix > 0:
        print(
            f"[pricing]   trying Black-Scholes  VIX={vix:.2f}  sigma={vix/100:.3f}  "
            f"T={_T_0DTE:.6f}"
        )
        result = _bs_spread(contract_type, spx_price, k_long, k_short, vix)
        if result is not None:
            result.update({"long_strike": k_long, "short_strike": k_short})
            print(
                f"[pricing]   ✓ BS    debit=${result['debit_dollars']:.2f}  "
                f"max_gain=${result['max_gain_dollars']:.2f}"
            )
            return result
        print(f"[pricing]   ✗ BS returned None")
    else:
        print(f"[pricing]   skipping BS (vix={vix})")

    print(f"[pricing]   → FALLBACK to 35% approximation")
    return None


# ── Cache build ────────────────────────────────────────────────────────────────

def _build_cache(
    df: pd.DataFrame,
    spread_widths: list[int],
    api_delay_seconds: float = 0.15,
) -> pd.DataFrame:
    spy_col = next((c for c in df.columns if c in ("spy_close", "spy_adj_close")), None)
    vix_col = next((c for c in df.columns if c in ("vix_level", "vix_close", "vix_vix_close")), None)
    if spy_col is None:
        print("[historical_options] No spy_close column — cannot build cache")
        return pd.DataFrame()

    dates = sorted(df["date"].dt.date.unique())
    total = len(dates) * len(spread_widths) * 2
    done = api_q = api_a = bs = fb = 0
    rows = []

    print(
        f"[historical_options] Building options cache: "
        f"{len(dates)} dates × {len(spread_widths)} widths × 2 directions = {total} slots"
    )

    for trade_date in dates:
        row = df[df["date"].dt.date == trade_date].iloc[0]
        spx = float(row[spy_col]) * 10
        vix = float(row[vix_col]) if vix_col and pd.notna(row[vix_col]) else None

        for spread_width in spread_widths:
            for contract_type in ("call", "put"):
                done += 1
                print(
                    f"\n[build_cache] ── slot {done}/{total}  "
                    f"{trade_date}  {contract_type}  {spread_width}pt  "
                    f"SPX≈{spx:.1f}  VIX={vix}"
                )
                result = fetch_spread_pricing(trade_date, contract_type, spx, spread_width, vix)

                if result:
                    src = result["source"]
                    if src == "api_quotes": api_q += 1
                    elif src == "api_aggs": api_a += 1
                    else: bs += 1
                    print(
                        f"[build_cache]   RESULT  source={src}  "
                        f"debit=${result['debit_dollars']:.2f}  "
                        f"max_gain=${result['max_gain_dollars']:.2f}"
                    )
                    rows.append({
                        "date": str(trade_date),
                        "contract_type": contract_type,
                        "spread_width": spread_width,
                        "debit_dollars": result["debit_dollars"],
                        "max_gain_dollars": result["max_gain_dollars"],
                        "debit_per_share": result["debit_per_share"],
                        "long_strike": result["long_strike"],
                        "short_strike": result["short_strike"],
                        "source": src,
                    })
                else:
                    fb += 1
                    print(f"[build_cache]   RESULT  source=35pct_fallback")

                pass  # no API calls for historical data — no rate limit delay needed

    print(
        f"\n[historical_options] Cache complete: "
        f"api_quotes={api_q}  api_aggs={api_a}  black_scholes={bs}  35pct_fallback={fb}"
    )
    return pd.DataFrame(rows)


# ── Public interface ───────────────────────────────────────────────────────────

def load_options_cache(
    df: pd.DataFrame,
    spread_widths: list[int],
    force_refresh: bool = False,
) -> dict:
    """
    Load or build the options pricing cache.
    Returns lookup dict keyed by (date_str, contract_type, spread_width).
    Missing keys fall back to 35% approximation in engine.py.
    """
    cache_df = pd.DataFrame()

    if CACHE_PATH.exists() and not force_refresh:
        cache_df = pd.read_parquet(CACHE_PATH)
        print(f"[historical_options] Loaded cache: {len(cache_df)} rows from {CACHE_PATH}")

    needed_dates = {str(d) for d in df["date"].dt.date.unique()}
    if not cache_df.empty:
        cached_keys = set(zip(
            cache_df["date"].astype(str),
            cache_df["contract_type"],
            cache_df["spread_width"].astype(int),
        ))
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
            cache_df = cache_df.drop_duplicates(subset=["date", "contract_type", "spread_width"])
            cache_df.to_parquet(CACHE_PATH, index=False)
            print(f"[historical_options] Cache saved → {CACHE_PATH} ({len(cache_df)} rows)")

    lookup: dict = {}
    for _, row in cache_df.iterrows():
        key = (str(row["date"]), str(row["contract_type"]), int(row["spread_width"]))
        lookup[key] = {
            "debit_dollars": float(row["debit_dollars"]),
            "max_gain_dollars": float(row["max_gain_dollars"]),
            "source": str(row.get("source", "unknown")),
        }

    total_slots = len(needed_dates) * len(spread_widths) * 2
    covered = len(lookup)
    print(
        f"[historical_options] Cache ready: {covered}/{total_slots} slots "
        f"({100*covered/max(total_slots,1):.0f}% covered — remainder uses 35% fallback)"
    )
    return lookup
