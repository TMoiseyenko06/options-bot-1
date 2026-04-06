"""
dashboard/signal_watcher.py — Continuous 0DTE signal watcher.

Runs 24/7.  At startup it loads historical daily data from Databento
(covers everything up to ~15 min ago), then subscribes to the live
1-minute SPY feed to fill the remaining gap.

Signal fires once per trading day when the 9:44am bar arrives
(delivered at 9:45am ET), using all accumulated data.

Usage:
    python dashboard/signal_watcher.py
    python dashboard/signal_watcher.py --force-directional
"""

import argparse
import json
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import xgboost as xgb
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from features.engineer import FEATURE_COLUMNS
from dashboard.morning_signal import (
    _fetch_es_live,
    _fetch_spy_live,
    _fetch_yf_last30,
    build_today_features,
    generate_signal,
    print_signal,
)
from dashboard.telegram import send_signal, send_error

DATABENTO_API_KEY = os.environ.get("DATABENTO_API_KEY")
MASSIVE_API_KEY = os.environ.get("MASSIVE_API_KEY")

MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
STATE_PATH = PROJECT_ROOT / "dashboard" / "watcher_state.json"

ET = ZoneInfo("America/New_York")

# Fire signal when the 9:44am bar arrives (it closes at 9:45am ET)
SIGNAL_HOUR = 9
SIGNAL_MINUTE = 44


class SignalWatcher:
    def __init__(self, force_directional: bool = False):
        self.force_directional = force_directional
        self.live_bars: list[dict] = []         # 1-min bars from live feed
        self.model: xgb.XGBClassifier | None = None
        self.feature_cols: list[str] = []
        self.existing_features: pd.DataFrame = pd.DataFrame()
        self.last_signal_date: str | None = None

        if STATE_PATH.exists():
            try:
                with open(STATE_PATH) as f:
                    state = json.load(f)
                self.last_signal_date = state.get("last_signal_date")
                print(f"[watcher] Last signal sent: {self.last_signal_date}")
            except Exception:
                pass

    # ── Startup ──────────────────────────────────────────────────────────────

    def load_model(self):
        model_path = MODELS_DIR / "best_model.json"
        feature_list_path = MODELS_DIR / "feature_list.json"
        print("[watcher] Loading model …")
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(model_path))
        with open(feature_list_path) as f:
            self.feature_cols = json.load(f)["features"]
        print(f"[watcher] Model loaded  ({len(self.feature_cols)} features)")

    def load_historical(self):
        features_path = PROCESSED_DIR / "features.parquet"
        if not features_path.exists():
            raise FileNotFoundError(
                f"[watcher] {features_path} not found — run pipeline first"
            )
        self.existing_features = pd.read_parquet(features_path)
        self.existing_features["date"] = pd.to_datetime(self.existing_features["date"])
        print(
            f"[watcher] Loaded {len(self.existing_features)} rows of historical features "
            f"({self.existing_features['date'].max().date()} most recent)"
        )

    # ── Live bar handling ─────────────────────────────────────────────────────

    def on_bar(self, bar: dict):
        """Called for each incoming 1-minute bar from the live feed."""
        self.live_bars.append(bar)

        # Keep at most 3 days of bars to avoid unbounded memory growth
        cutoff = datetime.now(tz=ET).date() - timedelta(days=3)
        self.live_bars = [b for b in self.live_bars if b["date"] >= cutoff]

        ts = bar["timestamp"]
        print(
            f"  [bar] {ts.strftime('%Y-%m-%d %H:%M ET')}  "
            f"SPY {bar['open']:.2f}/{bar['high']:.2f}/{bar['low']:.2f}/{bar['close']:.2f}"
        )

        # Trigger signal when the 9:44am bar arrives (closes at 9:45am)
        now_et = datetime.now(tz=ET)
        if (
            now_et.weekday() < 5                         # Mon–Fri only
            and bar["date"].weekday() < 5
            and bar["timestamp"].hour == SIGNAL_HOUR
            and bar["timestamp"].minute == SIGNAL_MINUTE
            and str(bar["date"]) != self.last_signal_date
        ):
            self._generate_and_send(bar["date"])

    def _bars_to_daily(self, trade_date: date) -> pd.DataFrame:
        """Aggregate live 1-min bars for trade_date into a single daily row."""
        day_bars = [b for b in self.live_bars if b["date"] == trade_date]
        if not day_bars:
            return pd.DataFrame()
        return pd.DataFrame([{
            "date": pd.Timestamp(trade_date),
            "open":   day_bars[0]["open"],
            "high":   max(b["high"]   for b in day_bars),
            "low":    min(b["low"]    for b in day_bars),
            "close":  day_bars[-1]["close"],
            "volume": sum(b["volume"] for b in day_bars),
        }])

    # ── Signal generation ─────────────────────────────────────────────────────

    def _generate_and_send(self, trade_date: date):
        today_str = str(trade_date)
        print(f"\n[watcher] ━━ Generating signal for {today_str} ━━")

        # ── Fetch historical daily context ───────────────────────────────────
        try:
            spy_live = _fetch_spy_live()
        except Exception as e:
            print(f"  WARNING: SPY historical fetch failed: {e}")
            spy_live = pd.DataFrame()

        try:
            es_live = _fetch_es_live()
        except Exception as e:
            print(f"  WARNING: ES historical fetch failed: {e}")
            es_live = pd.DataFrame()

        # ── Patch today's SPY row with live bar data (fills the 15-min gap) ──
        today_daily = self._bars_to_daily(trade_date)
        if not today_daily.empty:
            print(
                f"  [live bars] {len([b for b in self.live_bars if b['date'] == trade_date])} "
                f"bars for {today_str}  open={today_daily['open'].iloc[0]:.2f}  "
                f"close={today_daily['close'].iloc[0]:.2f}"
            )
            # Remove any stale today row from historical fetch then append live row
            if not spy_live.empty:
                spy_live = spy_live[spy_live["date"].dt.date != trade_date]
            spy_live = pd.concat([spy_live, today_daily], ignore_index=True)
        else:
            print(f"  [live bars] No bars for {today_str} — using historical only")

        # ── VIX + sectors via yfinance ────────────────────────────────────────
        try:
            vix_sectors = _fetch_yf_last30(["^VIX", "^VIX9D", "XLK", "XLF", "XLE", "XLV"])
            vix_cols = ["date"] + [c for c in vix_sectors.columns if "vix" in c.lower()]
            sector_cols = ["date"] + [
                c for c in vix_sectors.columns
                if any(t in c.lower() for t in ["xlk", "xlf", "xle", "xlv"])
            ]
            vix_live = vix_sectors[vix_cols] if len(vix_cols) > 1 else pd.DataFrame()
            sectors_live = vix_sectors[sector_cols] if len(sector_cols) > 1 else pd.DataFrame()
        except Exception as e:
            print(f"  WARNING: yfinance fetch failed: {e}")
            vix_live = pd.DataFrame()
            sectors_live = pd.DataFrame()

        # ── Fallbacks if fetches failed ───────────────────────────────────────
        if spy_live.empty:
            spy_live = self.existing_features[
                ["date", "spy_open", "spy_close", "spy_high", "spy_low"]
            ].tail(30).rename(columns={c: c.replace("spy_", "") for c in
                                       self.existing_features.columns if c != "date"
                                       and c.startswith("spy_")})

        if es_live.empty:
            es_cols = ["date"] + [c for c in self.existing_features.columns if c.startswith("es_")]
            if len(es_cols) > 1:
                es_live = self.existing_features[es_cols].tail(30).rename(
                    columns={c: c.replace("es_", "") for c in es_cols if c != "date"}
                )

        # ── Build features ────────────────────────────────────────────────────
        try:
            today_features = build_today_features(
                spy_live, es_live, vix_live, sectors_live, self.existing_features
            )
        except Exception as e:
            print(f"[watcher] ERROR building features: {e}")
            import traceback
            traceback.print_exc()
            send_error(f"Watcher feature build failed ({today_str}):\n{e}")
            return

        # ── Generate signal ───────────────────────────────────────────────────
        result = generate_signal(
            today_features, self.model, self.feature_cols,
            force_directional=self.force_directional,
        )

        # ── Options pricing ───────────────────────────────────────────────────
        result["options_pricing"] = None
        if result["signal"] != "NO TRADE" and MASSIVE_API_KEY:
            try:
                from dashboard.massive import get_trade_details
                direction = 1 if "CALL" in result["signal"] else -1
                spy_close_col = next(
                    (c for c in today_features.columns if "spy_close" in c.lower()), None
                )
                current_price = float(today_features[spy_close_col].values[0]) if spy_close_col else None
                if current_price:
                    result["options_pricing"] = get_trade_details(
                        direction=direction,
                        current_spx_price=current_price * 10,
                        spread_width=5,
                    )
            except Exception as e:
                print(f"  WARNING: Options pricing failed: {e}")

        # ── Output + send ─────────────────────────────────────────────────────
        print_signal(result)
        send_signal(result)

        sig_path = PROJECT_ROOT / "dashboard" / "last_signal.json"
        with open(sig_path, "w") as f:
            json.dump(result, f, indent=2, default=str)

        self.last_signal_date = today_str
        try:
            with open(STATE_PATH, "w") as f:
                json.dump({"last_signal_date": today_str}, f)
        except Exception:
            pass

        print(f"[watcher] Signal sent for {today_str}")

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        import databento as db

        self.load_model()
        self.load_historical()

        print(
            f"\n[watcher] Connecting to Databento live feed …\n"
            f"  Dataset : XNAS.ITCH\n"
            f"  Schema  : ohlcv-1m\n"
            f"  Symbol  : SPY\n"
            f"  Signal  : fires at {SIGNAL_HOUR:02d}:{SIGNAL_MINUTE+1:02d}am ET "
            f"(when 9:{SIGNAL_MINUTE:02d} bar closes)\n"
        )

        backoff = 5
        while True:
            try:
                live = db.Live(key=DATABENTO_API_KEY)
                live.subscribe(
                    dataset="XNAS.ITCH",
                    schema="ohlcv-1m",
                    symbols=["SPY"],
                    stype_in="raw_symbol",
                )
                print("[watcher] Live feed connected — waiting for bars …\n")
                backoff = 5  # reset on successful connection

                for record in live:
                    # OHLCV records have open/high/low/close/volume attributes
                    if not hasattr(record, "open"):
                        continue

                    # Timestamp: nanoseconds since epoch on hd.ts_event
                    ts_ns = getattr(getattr(record, "hd", None), "ts_event", None)
                    if ts_ns is None:
                        continue

                    ts_et = (
                        pd.Timestamp(ts_ns, unit="ns", tz="UTC")
                        .tz_convert("America/New_York")
                    )

                    # Databento fixed-point prices: divide by 1e9 for dollars
                    bar = {
                        "date":      ts_et.date(),
                        "timestamp": ts_et,
                        "open":      record.open   / 1_000_000_000,
                        "high":      record.high   / 1_000_000_000,
                        "low":       record.low    / 1_000_000_000,
                        "close":     record.close  / 1_000_000_000,
                        "volume":    record.volume,
                    }
                    self.on_bar(bar)

            except KeyboardInterrupt:
                print("\n[watcher] Shutting down gracefully.")
                break
            except Exception as e:
                print(f"[watcher] Feed error: {e}")
                print(f"[watcher] Reconnecting in {backoff}s …")
                time.sleep(backoff)
                backoff = min(backoff * 2, 120)  # exponential backoff, cap 2 min


def main():
    parser = argparse.ArgumentParser(description="Continuous 0DTE SPX signal watcher")
    parser.add_argument(
        "--force-directional", action="store_true",
        help="Bypass FLAT prediction and event filters (testing only)",
    )
    args = parser.parse_args()

    if not DATABENTO_API_KEY:
        print("[watcher] ERROR: DATABENTO_API_KEY not set in .env")
        sys.exit(1)

    watcher = SignalWatcher(force_directional=args.force_directional)
    watcher.run()


if __name__ == "__main__":
    main()
