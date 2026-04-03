"""
backtest/historical_options.py — Fetch and cache real historical SPX options pricing.

Fetches actual bid/ask quotes at 9:30am ET on each trade date from the Massive API.
This simulates 15-minute delayed data: the live feed at 9:45am (entry time) shows prices
from 9:30am, so backtesting uses 9:30am historical prices for identical consistency.

Results are cached to data/cache/options_prices.parquet to avoid repeat API calls.
When the API cannot return data for a date, callers fall back to the 35% approximation.
"""

import os
import time
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pytz
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
ET_TZ = pytz.timezone("America/New_York")

# 15-min delay: we fetch at 9:30am ET so the data matches what the live
# 15-min delayed feed would show at 9:45am (actual entry time).
_DELAYED_HOUR = 9
_DELAYED_MINUTE = 30


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
    """Build OCC-format options ticker, e.g. O:SPX260402C05220000."""
    exp_str = expiry.strftime("%y%m%d")
    c_type = "C" if contract_type.lower() == "call" else "P"
    strike_int = int(round(strike * 1000))
    return f"O:{underlying}{exp_str}{c_type}{str(strike_int).zfill(8)}"


# ── Strike selection ───────────────────────────────────────────────────────────

def _atm_strikes(
    contract_type: str, spx_price: float, spread_width: int
) -> tuple[float, float]:
    """
    Round spx_price to nearest 5-point increment to select first OTM strike.
    Mirrors the logic used in dashboard/massive.py for consistency.
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


# ── Quote / agg fetchers ───────────────────────────────────────────────────────

def _fetch_quote_at_930(occ_ticker: str, trade_date: date) -> Optional[dict]:
    """
    Fetch the last bid/ask quote at or before 9:30am ET via /v3/quotes.
    Returns the raw quote dict or None on failure.
    """
    dt_et = ET_TZ.localize(
        datetime(
            trade_date.year, trade_date.month, trade_date.day,
            _DELAYED_HOUR, _DELAYED_MINUTE, 0,
        )
    )
    # Polygon-style APIs accept nanosecond Unix timestamps
    timestamp_ns = int(dt_et.timestamp() * 1_000_000_000)
    try:
        data = _get(
            f"/v3/quotes/{occ_ticker}",
            {
                "timestamp.lte": timestamp_ns,
                "limit": 1,
                "order": "desc",
                "sort": "timestamp",
            },
        )
        results = data.get("results", [])
        return results[0] if results else None
    except Exception as exc:
        print(f"[historical_options]   Quote fetch failed for {occ_ticker}: {exc}")
        return None


def _fetch_agg_at_930(occ_ticker: str, trade_date: date) -> Optional[dict]:
    """
    Fallback: fetch 1-minute OHLCV bars and return the first bar at or after 9:30am ET.
    The close of that bar is used as a mid-price proxy.
    """
    date_str = trade_date.isoformat()
    try:
        data = _get(
            f"/v2/aggs/ticker/{occ_ticker}/range/1/minute/{date_str}/{date_str}",
            {"adjusted": "false", "sort": "asc", "limit": 50},
        )
        results = data.get("results", [])
        if not results:
            return None
        for bar in results:
            dt_et = (
                pytz.utc.localize(datetime.utcfromtimestamp(bar["t"] / 1000))
                .astimezone(ET_TZ)
            )
            if dt_et.hour > _DELAYED_HOUR or (
                dt_et.hour == _DELAYED_HOUR and dt_et.minute >= _DELAYED_MINUTE
            ):
                return bar
    except Exception as exc:
        print(f"[historical_options]   Agg fetch failed for {occ_ticker}: {exc}")
    return None


def _bid_ask_from_result(
    result: dict, source: str
) -> tuple[Optional[float], Optional[float]]:
    """Extract (bid, ask) from a quote or agg-bar result dict."""
    if source == "quote":
        bid = result.get("bid_price") or result.get("bid")
        ask = result.get("ask_price") or result.get("ask")
        return bid, ask
    # Agg bar: use close as both bid and ask (mid approximation)
    close = result.get("c")
    return close, close


# ── Single spread fetch ────────────────────────────────────────────────────────

def fetch_spread_pricing(
    trade_date: date,
    contract_type: str,
    spx_price: float,
    spread_width: int,
) -> Optional[dict]:
    """
    Fetch real historical spread pricing at 9:30am ET on trade_date.

    15-min delay simulation:
      The live feed at 9:45am (entry) shows data from 9:30am actual time.
      We fetch 9:30am historical prices to match that view exactly.

    Returns dict with debit_dollars / max_gain_dollars, or None if the API
    could not supply data for this date/contract (caller falls back to 35%).
    """
    long_strike, short_strike = _atm_strikes(contract_type, spx_price, spread_width)
    long_ticker = _occ_ticker("SPX", trade_date, contract_type, long_strike)
    short_ticker = _occ_ticker("SPX", trade_date, contract_type, short_strike)

    # Try /v3/quotes first; fall back to 1-min agg bars
    long_raw = _fetch_quote_at_930(long_ticker, trade_date)
    long_source = "quote"
    if long_raw is None:
        long_raw = _fetch_agg_at_930(long_ticker, trade_date)
        long_source = "agg"

    short_raw = _fetch_quote_at_930(short_ticker, trade_date)
    short_source = "quote"
    if short_raw is None:
        short_raw = _fetch_agg_at_930(short_ticker, trade_date)
        short_source = "agg"

    if long_raw is None or short_raw is None:
        return None

    _, long_ask = _bid_ask_from_result(long_raw, long_source)
    short_bid, _ = _bid_ask_from_result(short_raw, short_source)

    if long_ask is None or short_bid is None:
        return None

    debit_per_share = float(long_ask) - float(short_bid)
    if debit_per_share <= 0:
        return None

    debit_dollars = debit_per_share * SPX_MULTIPLIER
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
        "source": f"api_{long_source}/{short_source}",
    }


# ── Cache build ────────────────────────────────────────────────────────────────

def _build_cache(
    df: pd.DataFrame,
    spread_widths: list[int],
    api_delay_seconds: float = 0.25,
) -> pd.DataFrame:
    """
    Fetch historical pricing for all dates in df and return as a DataFrame.
    SPX price per date is approximated from the previous-day spy_close × 10.
    """
    spy_col = next(
        (c for c in df.columns if c in ("spy_close", "spy_adj_close")), None
    )
    if spy_col is None:
        print("[historical_options] No spy_close column — cannot fetch options data")
        return pd.DataFrame()

    dates = sorted(df["date"].dt.date.unique())
    total = len(dates) * len(spread_widths) * 2
    done = 0
    rows = []

    print(
        f"[historical_options] Fetching options prices: "
        f"{len(dates)} dates × {len(spread_widths)} widths × 2 directions "
        f"= {total} API requests"
    )

    for trade_date in dates:
        date_row = df[df["date"].dt.date == trade_date].iloc[0]
        spx_price = float(date_row[spy_col]) * 10  # SPY → SPX approximation

        for spread_width in spread_widths:
            for contract_type in ("call", "put"):
                done += 1
                if done % 20 == 0 or done == total:
                    print(f"[historical_options]   {done}/{total} fetched …")

                result = fetch_spread_pricing(
                    trade_date, contract_type, spx_price, spread_width
                )

                if result is not None:
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
                    print(
                        f"[historical_options]   No data for "
                        f"{trade_date} {contract_type} {spread_width}pt — "
                        f"will use 35% fallback"
                    )

                time.sleep(api_delay_seconds)

    return pd.DataFrame(rows)


# ── Public interface ───────────────────────────────────────────────────────────

def load_options_cache(
    df: pd.DataFrame,
    spread_widths: list[int],
    force_refresh: bool = False,
) -> dict:
    """
    Load the on-disk options pricing cache and supplement any missing dates
    by calling the Massive API.

    Returns a lookup dict keyed by (date_str, contract_type, spread_width):
        {
            ("2023-01-10", "call", 3): {"debit_dollars": 112.5, "max_gain_dollars": 187.5},
            ...
        }

    Dates not in the dict should fall back to the 35% approximation in engine.py.
    """
    if not MASSIVE_API_KEY:
        print(
            "[historical_options] MASSIVE_API_KEY not set — "
            "backtesting will use 35% debit approximation"
        )
        return {}

    cache_df = pd.DataFrame()

    if CACHE_PATH.exists() and not force_refresh:
        cache_df = pd.read_parquet(CACHE_PATH)
        print(
            f"[historical_options] Loaded cache: {len(cache_df)} rows "
            f"from {CACHE_PATH}"
        )

    # Find dates and (width, direction) combos missing from cache
    needed_dates = {str(d) for d in df["date"].dt.date.unique()}
    if not cache_df.empty:
        # A date is fully cached if all (width, direction) combos are present
        cached = set(
            zip(
                cache_df["date"].astype(str),
                cache_df["contract_type"],
                cache_df["spread_width"].astype(int),
            )
        )
        missing_dates = set()
        for d in needed_dates:
            for w in spread_widths:
                for ct in ("call", "put"):
                    if (d, ct, w) not in cached:
                        missing_dates.add(d)
                        break
    else:
        missing_dates = needed_dates

    if missing_dates:
        print(
            f"[historical_options] {len(missing_dates)} dates need fetching …"
        )
        missing_df = df[df["date"].dt.date.apply(str).isin(missing_dates)]
        new_data = _build_cache(missing_df, spread_widths)
        if not new_data.empty:
            cache_df = pd.concat([cache_df, new_data], ignore_index=True)
            # Deduplicate (in case of re-runs)
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

    api_count = len(lookup)
    total_slots = len(needed_dates) * len(spread_widths) * 2
    print(
        f"[historical_options] Options cache ready: "
        f"{api_count}/{total_slots} slots covered by real API data "
        f"({100*api_count/max(total_slots,1):.0f}%); "
        f"remainder will use 35% fallback"
    )
    return lookup
