# === FILE: utils.py ===
"""
utils.py – Tiện ích chung: timezone, session check, formatting, telegram.
"""
import pytz
import requests
from datetime import datetime
from config import TIMEZONE, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, ACTIVE_SESSIONS


_tz = pytz.timezone(TIMEZONE)


# ── Timezone helpers ───────────────────────────────────────────────────────────

def utc_to_vn(utc_dt: datetime) -> datetime:
    """Chuyển datetime UTC → VN (+07)."""
    if utc_dt.tzinfo is None:
        utc_dt = pytz.utc.localize(utc_dt)
    return utc_dt.astimezone(_tz)


def now_vn() -> datetime:
    return datetime.now(_tz)


def now_utc() -> datetime:
    return datetime.now(pytz.utc)


# ── Session helpers ────────────────────────────────────────────────────────────

def is_active_session(dt_utc: datetime | None = None) -> bool:
    """
    Kiểm tra có đang trong phiên giao dịch tốt không.
    Phiên tốt = Asian overlap, London, New York (giờ UTC).
    """
    if dt_utc is None:
        dt_utc = now_utc()
    h = dt_utc.hour
    return any(start <= h < end for start, end in ACTIVE_SESSIONS)


def session_name(dt_utc: datetime | None = None) -> str:
    if dt_utc is None:
        dt_utc = now_utc()
    h = dt_utc.hour
    if 1 <= h < 9:
        return 'Asian'
    if 7 <= h < 12:
        return 'London Open'
    if 12 <= h < 16:
        return 'London'
    if 13 <= h < 21:
        return 'New York'
    return 'Off-peak'


# ── Number formatting ──────────────────────────────────────────────────────────

def pct_str(v: float, decimals: int = 2) -> str:
    return f"{v:+.{decimals}f}%"


def price_str(v: float) -> str:
    return f"{v:,.2f}"


def slope_icon(v: float, threshold: float = 0.02) -> str:
    if v > threshold:
        return '↑'
    if v < -threshold:
        return '↓'
    return '↗'


# ── Telegram ───────────────────────────────────────────────────────────────────

def send_telegram(message: str) -> bool:
    """
    Gửi thông báo Telegram (nếu token được cấu hình).
    Trả về True nếu thành công.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        resp = requests.post(url, json={
            'chat_id': TELEGRAM_CHAT_ID,
            'text': message,
            'parse_mode': 'HTML'
        }, timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def format_signal_telegram(
    signal: str,
    close: float,
    prob_up: float,
    score_up: float,
    score_down: float,
    reason: str,
    vn_time: str,
) -> str:
    icon = '🟢 UP' if signal == 'UP' else '🔴 DOWN'
    return (
        f"<b>BTC Promax AI v5.0</b>\n"
        f"📌 Signal: {icon}\n"
        f"💰 Price: ${close:,.2f}\n"
        f"🤖 ML Prob UP: {prob_up:.3f}\n"
        f"📊 Score UP: {score_up:.1f} | DOWN: {score_down:.1f}\n"
        f"📝 Reason: {reason}\n"
        f"🕐 Time (VN): {vn_time}"
    )


# ── Misc ───────────────────────────────────────────────────────────────────────

def safe_float(value, default: float = 0.0) -> float:
    """Convert an arbitrary value to float safely."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def estimate_sentiment(news_list: list) -> tuple[float, str]:
    """Placeholder – thay bằng FinBERT nếu có."""
    return 0.0, 'neutral'
