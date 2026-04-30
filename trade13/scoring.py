# === FILE: scoring.py ===
"""
scoring.py – Tính điểm kỹ thuật UP/DOWN (0-10) từ đa chỉ báo.
v5.0: Bổ sung VWAP, ADX direction, candle patterns, market structure,
      S/R proximity, vol spike. Tổng trọng số = 10.
"""
import numpy as np
from utils import safe_float


# ── Weight table (tổng tối đa = 10.0) ────────────────────────────────────────
#  Trend  : 3.0  (slope 5m 1.2 | 15m 0.9 | 1h 0.6 | structure 0.3)
#  ADX    : 1.0
#  RSI    : 1.2
#  MACD   : 0.8
#  Stoch  : 0.7
#  BB     : 0.6
#  VWAP   : 0.7
#  Volume : 0.5
#  Pattern: 0.5
#  S/R    : 0.3
#  ATR    : 0.2

def compute_technical_score(row: dict | object) -> tuple[float, float]:
    """
    Tính (score_up, score_down) trong [0, 10].
    row có thể là dict hoặc pandas Series.
    """
    def g(key: str, default: float = 0.0) -> float:
        try:
            return safe_float(row[key], default)  # type: ignore[index]
        except (KeyError, TypeError):
            return default

    close         = g('close', 1.0)
    slope5        = g('ema20_slope_pct')
    slope15       = g('ema20_15m_slope_pct')
    slope1h       = g('ema20_1h_slope_pct')
    rsi           = g('rsi', 50.0)
    macd_diff     = g('macd_diff')
    di_diff       = g('di_diff')          # ADX +DI - (-DI)
    adx           = g('adx')
    stoch_k       = g('stoch_k', 50.0)
    stoch_d       = g('stoch_d', 50.0)
    bb_pct        = g('bb_pct', 0.5)
    atr_pct       = g('atr_pct')
    volume_ratio  = g('volume_ratio', 1.0)
    vwap_dist     = g('vwap_dist')        # % price vs VWAP
    structure     = g('structure')         # +1 bull / -1 bear / 0
    dist_res      = g('dist_to_resistance', 5.0)
    dist_sup      = g('dist_to_support', 5.0)
    pat_hammer    = g('pat_hammer')
    pat_star      = g('pat_shooting_star')
    pat_bull_eng  = g('pat_bull_engulf')
    pat_bear_eng  = g('pat_bear_engulf')
    rsi_div       = g('rsi_divergence')   # +1 bull / -1 bear
    vol_spike     = g('vol_spike')
    ema5_abv20    = g('ema5_above_20')
    ema10_abv20   = g('ema10_above_20')

    up   = 0.0
    down = 0.0

    # ── [1] Trend slope (3.0 total) ───────────────────────────────────────
    # 5m slope (1.2)
    if slope5 > 0.05:
        up   += 1.2
    elif slope5 > 0.025:
        up   += 0.7
    elif slope5 < -0.05:
        down += 1.2
    elif slope5 < -0.025:
        down += 0.7

    # 15m slope (0.9)
    if slope15 > 0.03:
        up   += 0.9
    elif slope15 > 0.015:
        up   += 0.5
    elif slope15 < -0.03:
        down += 0.9
    elif slope15 < -0.015:
        down += 0.5

    # 1h slope (0.6)
    if slope1h > 0.01:
        up   += 0.6
    elif slope1h < -0.01:
        down += 0.6

    # Market structure (0.3)
    if structure == 1:
        up   += 0.3
    elif structure == -1:
        down += 0.3

    # EMA cross bonus (included in trend)
    if ema5_abv20 and ema10_abv20:
        up   += 0.2
    elif not ema5_abv20 and not ema10_abv20:
        down += 0.2

    # ── [2] ADX (1.0) ─────────────────────────────────────────────────────
    if adx > 30:
        adx_bonus = 1.0
    elif adx > 22:
        adx_bonus = 0.6
    else:
        adx_bonus = 0.0

    if di_diff > 5:      # +DI dominant → bullish pressure
        up   += adx_bonus
    elif di_diff < -5:   # -DI dominant → bearish pressure
        down += adx_bonus

    # ── [3] RSI (1.2) ─────────────────────────────────────────────────────
    if rsi < 25:
        up   += 1.2
    elif rsi < 35:
        up   += 0.7
    elif rsi < 45:
        up   += 0.3
    if rsi > 75:
        down += 1.2
    elif rsi > 65:
        down += 0.7
    elif rsi > 55:
        down += 0.3

    # RSI divergence bonus
    if rsi_div == 1:
        up   += 0.3
    elif rsi_div == -1:
        down += 0.3

    # ── [4] MACD (0.8) ────────────────────────────────────────────────────
    if macd_diff > 0:
        strength = min(macd_diff / 50, 1.0)
        if slope5 > 0:
            up += 0.8 * strength
        else:
            up += 0.3 * strength
    elif macd_diff < 0:
        strength = min(abs(macd_diff) / 50, 1.0)
        if slope5 < 0:
            down += 0.8 * strength
        else:
            down += 0.3 * strength

    # ── [5] Stochastic (0.7) ──────────────────────────────────────────────
    if stoch_k < 20 and stoch_k > stoch_d:
        up   += 0.7
    elif stoch_k < 30 and stoch_k > stoch_d:
        up   += 0.4
    if stoch_k > 80 and stoch_k < stoch_d:
        down += 0.7
    elif stoch_k > 70 and stoch_k < stoch_d:
        down += 0.4

    # ── [6] Bollinger Bands (0.6) ─────────────────────────────────────────
    if bb_pct < 0.1:
        up   += 0.6
    elif bb_pct < 0.25:
        up   += 0.3
    if bb_pct > 0.9:
        down += 0.6
    elif bb_pct > 0.75:
        down += 0.3

    # ── [7] VWAP (0.7) ────────────────────────────────────────────────────
    if vwap_dist > 0.15:       # giá trên VWAP → bullish bias
        up   += 0.5
    elif vwap_dist > 0.05:
        up   += 0.25
    elif vwap_dist < -0.15:    # giá dưới VWAP → bearish bias
        down += 0.5
    elif vwap_dist < -0.05:
        down += 0.25
    # Quay lại VWAP từ dưới
    if -0.05 < vwap_dist < 0.05 and slope5 > 0:
        up   += 0.2
    elif -0.05 < vwap_dist < 0.05 and slope5 < 0:
        down += 0.2

    # ── [8] Volume (0.5) ──────────────────────────────────────────────────
    if volume_ratio > 1.5:
        if slope5 > 0:
            up   += 0.5
        elif slope5 < 0:
            down += 0.5
    elif volume_ratio > 1.2:
        if slope5 > 0:
            up   += 0.25
        elif slope5 < 0:
            down += 0.25
    if vol_spike and slope5 > 0:
        up   += 0.2
    elif vol_spike and slope5 < 0:
        down += 0.2

    # ── [9] Candle patterns (0.5) ─────────────────────────────────────────
    if pat_hammer:
        up   += 0.4
    if pat_bull_eng:
        up   += 0.5
    if pat_star:
        down += 0.4
    if pat_bear_eng:
        down += 0.5

    # ── [10] S/R Proximity (0.3) ──────────────────────────────────────────
    if dist_sup < 0.3:          # gần support → có thể bật lên
        up   += 0.3
    if dist_res < 0.3:          # gần resistance → có thể giảm
        down += 0.3

    # ── [11] ATR stability (0.2) ──────────────────────────────────────────
    if atr_pct < 0.12:
        if up > down:
            up   += 0.2
        else:
            down += 0.2

    score_up   = float(np.clip(up,   0, 10))
    score_down = float(np.clip(down, 0, 10))
    return score_up, score_down


def score_confidence(score: float, threshold: float) -> str:
    """Trả về nhãn confidence dựa trên khoảng cách score so với threshold."""
    margin = score - threshold
    if margin >= 2.5:
        return 'VERY_HIGH'
    if margin >= 1.5:
        return 'HIGH'
    if margin >= 0.5:
        return 'MEDIUM'
    return 'LOW'
