"""
dashboard/telegram.py — Send signal alerts to a Telegram bot.

Requires in .env:
    TELEGRAM_BOT_TOKEN=123456:ABC-your-token
    TELEGRAM_CHAT_ID=your_chat_id

How to set up:
    1. Message @BotFather on Telegram → /newbot → copy the token
    2. Message your new bot once, then visit:
       https://api.telegram.org/bot<token>/getUpdates
       Find "chat":{"id": <number>} — that is your TELEGRAM_CHAT_ID
    3. Add both values to your .env file
"""

import os
import requests

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
_TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


def _is_configured() -> bool:
    return bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)


def _send(text: str, parse_mode: str = "HTML") -> bool:
    """
    Send a message via the Telegram Bot API.
    Returns True on success, False on failure (never raises).
    """
    if not _is_configured():
        print("[telegram] Not configured — set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env")
        return False

    url = _TELEGRAM_API.format(token=TELEGRAM_BOT_TOKEN)
    try:
        resp = requests.post(
            url,
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": parse_mode},
            timeout=10,
        )
        if resp.status_code == 200:
            return True
        print(f"[telegram] API error {resp.status_code}: {resp.text}")
        return False
    except Exception as e:
        print(f"[telegram] Failed to send message: {e}")
        return False


def format_signal_message(result: dict) -> str:
    """Format the signal result dict as a clean HTML Telegram message."""
    sig = result["signal"]
    date = result["date"]
    confidence = result["confidence"]

    # Signal emoji
    if sig == "NO TRADE":
        emoji = "⏸"
    elif "CALL" in sig:
        emoji = "🟢"
    else:
        emoji = "🔴"

    lines = [
        f"{emoji} <b>0DTE SPX SIGNAL — {date}</b>",
        "",
        f"<b>Signal:</b> {sig}",
        f"<b>Confidence:</b> {confidence:.1%}",
    ]

    if result.get("signal_overridden"):
        lines.append(f"<i>(Raw: {result['raw_signal']} — overridden by filter)</i>")

    # Probabilities
    lines.append("")
    lines.append("<b>Probabilities:</b>")
    for label, prob in result["probabilities"].items():
        bar = "█" * int(prob * 10) + "░" * (10 - int(prob * 10))
        lines.append(f"  {label}: {prob:.1%} {bar}")

    # Top SHAP features
    lines.append("")
    lines.append("<b>Top drivers:</b>")
    for row in result["top_shap_features"][:5]:
        arrow = "▲" if row["shap_value"] > 0 else "▼"
        lines.append(
            f"  {arrow} {row['feature']}: {row['feature_value']:.4f} "
            f"(SHAP {row['shap_value']:+.3f})"
        )

    # Market context
    lines.append("")
    lines.append("<b>Market context:</b>")
    if result.get("vix") is not None:
        lines.append(f"  VIX: {result['vix']:.2f}")
    if result.get("vix9d_minus_vix") is not None:
        spread = result["vix9d_minus_vix"]
        term = "inverted" if spread < 0 else "normal"
        lines.append(f"  VIX9D-VIX: {spread:.2f} ({term})")
    lines.append(f"  Days to FOMC: {result['days_to_fomc']}")
    lines.append(f"  Days to CPI: {result['days_to_cpi']}")

    # Active filters / warnings
    if result.get("active_filters"):
        lines.append("")
        lines.append("<b>Active filters:</b>")
        for f in result["active_filters"]:
            lines.append(f"  ⚠ {f}")

    # Options trade ticket from Massive API
    p = result.get("options_pricing")
    if p:
        lines.append("")
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        lines.append(f"<b>TRADE TICKET</b>")
        price_str = f"@ {p['underlying_price']:.2f}" if p['underlying_price'] is not None else "(price unavailable)"
        lines.append(f"  {p['underlying']} {price_str}")
        lines.append(f"  {p['contract_type'].upper()} DEBIT SPREAD  |  0DTE")
        lines.append(f"  Long:  <code>{p['long_strike']:.0f}</code>  ({p['long_ticker']})")
        lines.append(f"  Short: <code>{p['short_strike']:.0f}</code>  ({p['short_ticker']})")
        lines.append(f"  Entry: 9:45am ET")
        if p.get("debit_dollars") is not None:
            lines.append("")
            lines.append(f"  Debit:     <b>${p['debit_dollars']:.2f}</b>")
            lines.append(f"  Max gain:  <b>${p['max_gain_dollars']:.2f}</b>")
            lines.append(f"  Max loss:  <b>${p['max_loss_dollars']:.2f}</b>")
            lines.append(f"  Breakeven: <b>{p['breakeven']:.2f}</b>")
            rr = p['max_gain_dollars'] / p['max_loss_dollars'] if p['max_loss_dollars'] else 0
            lines.append(f"  R/R:       {rr:.2f}x")
        if p.get("long_bid") is not None:
            lines.append("")
            lines.append(f"  Long  b/a: {p['long_bid']:.2f} / {p['long_ask']:.2f}")
            lines.append(f"  Short b/a: {p['short_bid']:.2f} / {p['short_ask']:.2f}")
        if p.get("long_iv"):
            lines.append(f"  Long IV:   {p['long_iv']:.1%}")
        if p.get("net_delta") is not None:
            lines.append(f"  Net delta: {p['net_delta']:+.4f}")
            lines.append(f"  Net theta: {p['net_theta']:+.4f}/day")
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    return "\n".join(lines)


def send_signal(result: dict) -> bool:
    """Format and send the signal result to Telegram. Returns True on success."""
    if not _is_configured():
        return False
    message = format_signal_message(result)
    ok = _send(message)
    if ok:
        print(f"[telegram] Signal sent to chat {TELEGRAM_CHAT_ID}")
    return ok


def send_error(error_msg: str) -> bool:
    """Send an error alert to Telegram."""
    if not _is_configured():
        return False
    return _send(f"⚠ <b>Signal Error</b>\n\n{error_msg}")
