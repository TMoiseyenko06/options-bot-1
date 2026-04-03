"""
dashboard/massive.py — Fetch real SPX options pricing via the Massive API.

Used only from morning_signal.py. Requires MASSIVE_API_KEY in .env.

Workflow:
  1. Get today's 0DTE SPX contracts near the money
  2. Find first OTM strike for the signal direction
  3. Fetch snapshots for both legs of the spread
  4. Return a complete spread ticket: strikes, debit, max gain,
     breakeven, net Greeks, IV
"""

import os
from datetime import date
from typing import Optional

import requests

MASSIVE_API_KEY = os.environ.get("MASSIVE_API_KEY")
MASSIVE_BASE_URL = os.environ.get("MASSIVE_BASE_URL", "https://api.massive.com")

# SPX option multiplier: each contract = $100 per index point
SPX_MULTIPLIER = 100


def _headers() -> dict:
    if not MASSIVE_API_KEY:
        raise EnvironmentError(
            "[massive] MASSIVE_API_KEY not set in .env"
        )
    return {"Authorization": f"Bearer {MASSIVE_API_KEY}"}


def _get(path: str, params: dict = None) -> dict:
    """Make a GET request to the Massive API. Raises on non-200."""
    url = f"{MASSIVE_BASE_URL}{path}"
    resp = requests.get(url, headers=_headers(), params=params or {}, timeout=10)
    if resp.status_code != 200:
        raise RuntimeError(
            f"[massive] API error {resp.status_code} on {path}: {resp.text[:200]}"
        )
    return resp.json()


# ── OCC ticker construction ────────────────────────────────────────────────────

def _occ_ticker(underlying: str, expiry: date, contract_type: str, strike: float) -> str:
    """
    Build an OCC-format options ticker.
    Example: O:SPX260402C05220000
      underlying  = SPX
      expiry      = 2026-04-02  → 260402
      type        = C or P
      strike      = 5220.0      → 05220000 (strike * 1000, 8 digits, zero-padded)
    """
    exp_str = expiry.strftime("%y%m%d")
    c_type = "C" if contract_type.lower() == "call" else "P"
    strike_int = int(round(strike * 1000))
    strike_str = str(strike_int).zfill(8)
    return f"O:{underlying}{exp_str}{c_type}{strike_str}"


# ── Contract lookup ────────────────────────────────────────────────────────────

def get_0dte_strikes(
    underlying: str,
    contract_type: str,
    current_price: float,
    spread_width: int,
    expiry: Optional[date] = None,
) -> tuple[float, float]:
    """
    Find the first OTM strike and the short strike (first OTM + spread_width).

    For a call spread: long = first strike >= current_price (first OTM call)
                       short = long + spread_width
    For a put spread:  long = first strike <= current_price (first OTM put)
                       short = long - spread_width

    Returns (long_strike, short_strike).
    """
    if expiry is None:
        expiry = date.today()

    # Fetch contracts near the money (within 50 points either side)
    if contract_type.lower() == "call":
        params = {
            "underlying_ticker": underlying,
            "contract_type": "call",
            "expiration_date": expiry.isoformat(),
            "strike_price.gte": current_price,
            "strike_price.lte": current_price + 50,
            "order": "asc",
            "sort": "strike_price",
            "limit": 20,
        }
    else:
        params = {
            "underlying_ticker": underlying,
            "contract_type": "put",
            "expiration_date": expiry.isoformat(),
            "strike_price.gte": current_price - 50,
            "strike_price.lte": current_price,
            "order": "desc",
            "sort": "strike_price",
            "limit": 20,
        }

    data = _get("/v3/reference/options/contracts", params)
    results = data.get("results", [])

    if not results:
        # Fallback: round to nearest 5 and compute manually
        if contract_type.lower() == "call":
            long_strike = round(current_price / 5) * 5
            if long_strike <= current_price:
                long_strike += 5
        else:
            long_strike = round(current_price / 5) * 5
            if long_strike >= current_price:
                long_strike -= 5
        short_strike = long_strike + spread_width if contract_type == "call" else long_strike - spread_width
        print(f"[massive] No contracts found via API — using rounded strikes: {long_strike}/{short_strike}")
        return float(long_strike), float(short_strike)

    long_strike = float(results[0]["strike_price"])
    short_strike = long_strike + spread_width if contract_type.lower() == "call" else long_strike - spread_width
    return long_strike, short_strike


# ── Snapshot fetch ─────────────────────────────────────────────────────────────

def get_contract_snapshot(underlying: str, occ_ticker: str) -> dict:
    """
    Fetch a full snapshot for one contract.
    Returns the results dict directly.
    """
    path = f"/v3/snapshot/options/{underlying}/{occ_ticker}"
    data = _get(path)
    return data.get("results", {})


# ── Spread pricing ─────────────────────────────────────────────────────────────

def get_spread_pricing(
    underlying: str,
    contract_type: str,
    long_strike: float,
    short_strike: float,
    expiry: Optional[date] = None,
    spread_width: Optional[int] = None,
) -> dict:
    """
    Fetch both legs of the spread and compute net pricing.

    For a debit spread:
      - Long leg: we PAY the ask
      - Short leg: we RECEIVE the bid
      - Net debit = long_ask - short_bid  (per share equivalent)
      - Net debit dollars = debit * SPX_MULTIPLIER
      - Max gain dollars = (spread_width * SPX_MULTIPLIER) - debit_dollars
      - Breakeven = long_strike + debit  (calls)
                  = long_strike - debit  (puts)

    Returns a dict with all trade details.
    """
    if expiry is None:
        expiry = date.today()

    long_ticker = _occ_ticker(underlying, expiry, contract_type, long_strike)
    short_ticker = _occ_ticker(underlying, expiry, contract_type, short_strike)

    print(f"[massive] Fetching long leg  : {long_ticker}")
    long_snap = get_contract_snapshot(underlying, long_ticker)

    print(f"[massive] Fetching short leg : {short_ticker}")
    short_snap = get_contract_snapshot(underlying, short_ticker)

    # Extract quotes
    long_quote = long_snap.get("last_quote", {})
    short_quote = short_snap.get("last_quote", {})

    long_ask = long_quote.get("ask")
    long_bid = long_quote.get("bid")
    long_mid = long_quote.get("midpoint")

    short_ask = short_quote.get("ask")
    short_bid = short_quote.get("bid")
    short_mid = short_quote.get("midpoint")

    # Use midpoint if bid/ask not available
    long_price = long_ask if long_ask is not None else long_mid
    short_price = short_bid if short_bid is not None else short_mid

    # Compute spread pricing
    if long_price is not None and short_price is not None:
        debit_per_share = long_price - short_price
        debit_dollars = debit_per_share * SPX_MULTIPLIER
    else:
        debit_per_share = None
        debit_dollars = None

    actual_width = abs(long_strike - short_strike)
    max_value_dollars = actual_width * SPX_MULTIPLIER if spread_width is None else spread_width * SPX_MULTIPLIER

    if debit_dollars is not None:
        max_gain_dollars = max_value_dollars - debit_dollars
        max_loss_dollars = debit_dollars
        if contract_type.lower() == "call":
            breakeven = long_strike + debit_per_share
        else:
            breakeven = long_strike - debit_per_share
    else:
        max_gain_dollars = None
        max_loss_dollars = None
        breakeven = None

    # Net Greeks (long - short for debit spread)
    long_greeks = long_snap.get("greeks", {})
    short_greeks = short_snap.get("greeks", {})

    def net_greek(key):
        l = long_greeks.get(key)
        s = short_greeks.get(key)
        if l is not None and s is not None:
            return round(l - s, 6)
        return None

    # Current underlying price
    underlying_price = (
        long_snap.get("underlying_asset", {}).get("price")
        or short_snap.get("underlying_asset", {}).get("price")
    )

    # IV (long leg IV is most relevant for cost)
    long_iv = long_snap.get("implied_volatility")
    short_iv = short_snap.get("implied_volatility")

    return {
        "underlying": underlying,
        "underlying_price": underlying_price,
        "contract_type": contract_type.lower(),
        "expiry": expiry.isoformat(),
        "long_strike": long_strike,
        "short_strike": short_strike,
        "long_ticker": long_ticker,
        "short_ticker": short_ticker,
        # Pricing
        "long_bid": long_bid,
        "long_ask": long_ask,
        "long_mid": long_mid,
        "short_bid": short_bid,
        "short_ask": short_ask,
        "short_mid": short_mid,
        "debit_per_share": round(debit_per_share, 4) if debit_per_share is not None else None,
        "debit_dollars": round(debit_dollars, 2) if debit_dollars is not None else None,
        "max_gain_dollars": round(max_gain_dollars, 2) if max_gain_dollars is not None else None,
        "max_loss_dollars": round(max_loss_dollars, 2) if max_loss_dollars is not None else None,
        "breakeven": round(breakeven, 2) if breakeven is not None else None,
        "max_value_dollars": max_value_dollars,
        # Greeks (net spread)
        "net_delta": net_greek("delta"),
        "net_gamma": net_greek("gamma"),
        "net_theta": net_greek("theta"),
        "net_vega": net_greek("vega"),
        # IV
        "long_iv": round(long_iv, 4) if long_iv else None,
        "short_iv": round(short_iv, 4) if short_iv else None,
        # Open interest
        "long_oi": long_snap.get("open_interest"),
        "short_oi": short_snap.get("open_interest"),
    }


# ── Main entry point ───────────────────────────────────────────────────────────

def get_trade_details(
    direction: int,
    current_spx_price: float,
    spread_width: int = 5,
    expiry: Optional[date] = None,
) -> Optional[dict]:
    """
    Top-level function called from morning_signal.py.

    direction: 1 = call spread, -1 = put spread
    Returns full spread pricing dict, or None if API fails.
    """
    if not MASSIVE_API_KEY:
        print("[massive] MASSIVE_API_KEY not set — skipping options pricing")
        return None

    if expiry is None:
        expiry = date.today()

    contract_type = "call" if direction == 1 else "put"

    try:
        long_strike, short_strike = get_0dte_strikes(
            "SPX", contract_type, current_spx_price, spread_width, expiry
        )
        pricing = get_spread_pricing(
            "SPX", contract_type, long_strike, short_strike, expiry, spread_width
        )
        return pricing
    except Exception as e:
        print(f"[massive] Failed to fetch options pricing: {e}")
        return None
