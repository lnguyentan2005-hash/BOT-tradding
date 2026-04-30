# === FILE: logger.py ===
"""
logger.py – Ghi log giao dịch ra CSV (live và backtest).
v5.0: Fix Score_Up/Down, thêm đầy đủ columns, hỗ trợ 2 file log riêng biệt.
"""
import csv
import os
from datetime import datetime

import pytz

from config import TRADE_LOG_CSV, BACKTEST_LOG_CSV, TIMEZONE
from scoring import compute_technical_score

_tz = pytz.timezone(TIMEZONE)

# ── Column schema ─────────────────────────────────────────────────────────────
FIELDNAMES = [
    # Kết quả giao dịch — ĐẶT ĐẦU để dễ lọc
    'Time_VN', 'Candle_VN',
    'Decision', 'Result', 'PnL',
    'Reason',
    # Price
    'Open', 'High', 'Low', 'Close', 'Volume',
    # ML probabilities
    'ML_Avg_Prob', 'LSTM_Prob', 'LGB_Prob', 'XGB_Prob',
    'ML_votes_up', 'ML_votes_down',
    # Technical scores
    'Score_Up', 'Score_Down',
    # Trend
    'UP_votes', 'DOWN_votes',
    # Indicators
    'EMA5', 'EMA10', 'EMA20', 'EMA20_slope_5m',
    'EMA20_slope_15m', 'EMA20_slope_30m', 'EMA20_slope_1h',
    'RSI', 'ADX', 'DI_diff', 'ATR_pct', 'ATR_ratio',
    'MACD_diff', 'Stoch_K', 'Stoch_D', 'BB_pct', 'BB_squeeze',
    'VWAP_dist', 'Volume_ratio', 'Vol_spike',
    'Structure', 'Pattern',
    'Dist_resistance', 'Dist_support',
]

_PROFIT_WIN  =  0.95   # profit per win unit
_PROFIT_LOSS = -1.00   # loss per loss unit


def init_log(path: str = TRADE_LOG_CSV) -> None:
    """Tạo file CSV với header nếu chưa tồn tại."""
    if not os.path.exists(path):
        with open(path, 'w', newline='', encoding='utf-8-sig') as f:
            csv.writer(f).writerow(FIELDNAMES)


def _candle_vn(row: dict | object) -> str:
    try:
        ts = row['timestamp']           # type: ignore[index]
        if hasattr(ts, 'tz_convert'):
            return ts.tz_convert(_tz).strftime('%Y-%m-%d %H:%M:%S')
        if hasattr(ts, 'strftime'):
            return ts.strftime('%Y-%m-%d %H:%M:%S')
    except Exception:
        pass
    return ''


def _get(row: dict | object, key: str, default: str = '') -> str:
    try:
        v = row[key]                    # type: ignore[index]
        if v is None or (isinstance(v, float) and v != v):
            return default
        return str(round(float(v), 6)) if isinstance(v, (int, float)) else str(v)
    except (KeyError, TypeError):
        return default


def _pattern_label(row: dict | object) -> str:
    """Tóm tắt pattern nến thành chuỗi ngắn."""
    patterns = []
    try:
        if row.get('pat_hammer'):              patterns.append('Hammer')       # type: ignore[attr-defined]
        if row.get('pat_shooting_star'):       patterns.append('ShootingStar')
        if row.get('pat_bull_engulf'):         patterns.append('BullEngulf')
        if row.get('pat_bear_engulf'):         patterns.append('BearEngulf')
        if row.get('pat_doji'):                patterns.append('Doji')
    except AttributeError:
        pass
    return '+'.join(patterns) if patterns else 'None'


def log_trade(
    row: dict | object,
    ml_prob: float | None,
    ml_details: dict[str, float] | None,
    decision: str,
    reason: str,
    result: str = '',
    path: str = TRADE_LOG_CSV,
) -> None:
    """
    Ghi 1 dòng log vào CSV.
    result: 'WIN' | 'LOSS' | '' (chưa có kết quả)
    """
    now_vn = datetime.now(_tz).strftime('%Y-%m-%d %H:%M:%S')

    # Score
    score_up, score_down = compute_technical_score(row)

    # ML details
    lstm_p = lgb_p = xgb_p = ''
    votes_up = votes_dn = ''
    if ml_details:
        lstm_p   = round(ml_details.get('lstm', 0), 4)
        lgb_p    = round(ml_details.get('lgbm', 0), 4)
        xgb_p    = round(ml_details.get('xgb',  0), 4)
        votes_up = sum(1 for p in ml_details.values() if p > 0.5)
        votes_dn = len(ml_details) - votes_up

    # Trend votes
    slopes = [
        _get(row, 'ema20_slope_pct', '0'),
        _get(row, 'ema20_15m_slope_pct', '0'),
        _get(row, 'ema20_30m_slope_pct', '0'),
        _get(row, 'ema20_1h_slope_pct', '0'),
    ]
    try:
        up_v  = sum(1 for s in slopes if float(s) > 0)
        dn_v  = sum(1 for s in slopes if float(s) < 0)
    except ValueError:
        up_v = dn_v = ''

    # PnL
    pnl = ''
    if result == 'WIN':
        pnl = _PROFIT_WIN
    elif result == 'LOSS':
        pnl = _PROFIT_LOSS

    entry = [
        now_vn,
        _candle_vn(row),
        decision, result, pnl,
        reason[:120],
        _get(row, 'open'), _get(row, 'high'), _get(row, 'low'), _get(row, 'close'), _get(row, 'volume'),
        round(ml_prob, 4) if ml_prob is not None else '',
        lstm_p, lgb_p, xgb_p,
        votes_up, votes_dn,
        round(score_up, 2), round(score_down, 2),
        up_v, dn_v,
        _get(row, 'ema5'), _get(row, 'ema10'), _get(row, 'ema20'),
        _get(row, 'ema20_slope_pct'),
        _get(row, 'ema20_15m_slope_pct'),
        _get(row, 'ema20_30m_slope_pct'),
        _get(row, 'ema20_1h_slope_pct'),
        _get(row, 'rsi'), _get(row, 'adx'), _get(row, 'di_diff'), _get(row, 'atr_pct'), _get(row, 'atr_ratio'),
        _get(row, 'macd_diff'), _get(row, 'stoch_k'), _get(row, 'stoch_d'), _get(row, 'bb_pct'), _get(row, 'bb_squeeze'),
        _get(row, 'vwap_dist'), _get(row, 'volume_ratio'), _get(row, 'vol_spike'),
        _get(row, 'structure'),
        _pattern_label(row),
        _get(row, 'dist_to_resistance'), _get(row, 'dist_to_support'),
    ]

    with open(path, 'a', newline='', encoding='utf-8-sig') as f:
        csv.writer(f).writerow(entry)


# ── Backtest log (riêng) ──────────────────────────────────────────────────────

def init_backtest_log() -> None:
    init_log(BACKTEST_LOG_CSV)


def log_backtest_trade(
    row: dict | object,
    ml_prob: float | None,
    ml_details: dict[str, float] | None,
    decision: str,
    reason: str,
    result: str,
) -> None:
    log_trade(
        row=row,
        ml_prob=ml_prob,
        ml_details=ml_details,
        decision=decision,
        reason=reason,
        result=result,
        path=BACKTEST_LOG_CSV,
    )
