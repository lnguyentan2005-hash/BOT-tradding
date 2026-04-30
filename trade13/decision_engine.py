# === FILE: decision_engine.py ===
"""
decision_engine.py – Ra quyết định UP / DOWN / SKIP với logic nhiều lớp.
v5.0: Thêm session filter, dynamic threshold, volatility check,
      weighted trend consensus, DI direction confirmation.
"""
from __future__ import annotations
import numpy as np
from datetime import datetime
import pytz

from config import (
    ADX_THRESHOLD, SLOPE_THRESHOLD, MIN_TREND_FRAMES,
    CONSENSUS_RATIO, ML_PROB_THRESHOLD, ML_HIGH_CONF,
    TECH_SCORE_THRESHOLD, TECH_SCORE_CAUTIOUS,
    MAX_CONSECUTIVE_LOSSES, CAUTIOUS_CANDLES,
    VOLATILITY_MAX_ATR, DAILY_LOSS_LIMIT, TIMEZONE,
)
from scoring import compute_technical_score, score_confidence
from utils import is_active_session, safe_float


class RiskState:
    """Quản lý trạng thái rủi ro trong phiên."""

    def __init__(self) -> None:
        self.consecutive_losses : int  = 0
        self.cautious_candles_left: int = 0
        self.daily_losses        : int  = 0
        self.daily_wins          : int  = 0
        self._last_reset_day     : int  = -1

    def _maybe_reset_daily(self) -> None:
        today = datetime.now(pytz.timezone(TIMEZONE)).day
        if today != self._last_reset_day:
            self.daily_losses    = 0
            self.daily_wins      = 0
            self._last_reset_day = today

    def record_result(self, won: bool) -> None:
        self._maybe_reset_daily()
        if won:
            self.consecutive_losses = 0
            self.cautious_candles_left = 0
            self.daily_wins += 1
        else:
            self.consecutive_losses += 1
            self.daily_losses       += 1
            if self.consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
                self.cautious_candles_left = CAUTIOUS_CANDLES

    def tick(self) -> None:
        """Gọi mỗi nến (5m) để giảm cautious countdown."""
        if self.cautious_candles_left > 0:
            self.cautious_candles_left -= 1

    @property
    def is_cautious(self) -> bool:
        return self.cautious_candles_left > 0

    @property
    def is_daily_limit_hit(self) -> bool:
        self._maybe_reset_daily()
        return self.daily_losses >= DAILY_LOSS_LIMIT


# ── Weighted slope consensus ──────────────────────────────────────────────────

_TF_WEIGHTS = {
    '5m' : 1.0,
    '15m': 1.5,   # 15m / 30m ảnh hưởng nhiều hơn 5m noise
    '30m': 2.0,
    '1h' : 2.5,
}

def _weighted_slope_vote(row: dict | object) -> tuple[float, float]:
    """
    Trả về (weighted_up_score, weighted_down_score).
    Mỗi khung đóng góp trọng số khác nhau.
    """
    slopes = {
        '5m' : safe_float(row['ema20_slope_pct']),         # type: ignore[index]
        '15m': safe_float(row.get('ema20_15m_slope_pct', 0)),  # type: ignore[attr-defined]
        '30m': safe_float(row.get('ema20_30m_slope_pct', 0)),
        '1h' : safe_float(row.get('ema20_1h_slope_pct', 0)),
    }
    up_w = down_w = 0.0
    for tf, slope in slopes.items():
        w = _TF_WEIGHTS[tf]
        if slope > 0:
            up_w   += w
        elif slope < 0:
            down_w += w
    return up_w, down_w


# ── Main decision function ────────────────────────────────────────────────────

def make_decision(
    row: dict | object,
    ml_prob_up: float | None,
    ml_details: dict[str, float] | None,
    risk_state: RiskState,
) -> tuple[str, str]:
    """
    Pipeline quyết định: SKIP / UP / DOWN.
    Trả về (signal, reason).
    Mỗi lớp SKIP có lý do rõ ràng để dễ debug và phân tích log.
    """

    # ── L0: Daily loss limit ─────────────────────────────────────────────
    if risk_state.is_daily_limit_hit:
        return 'SKIP', f"Daily loss limit ({DAILY_LOSS_LIMIT}) hit — rest today"

    # ── L1: Volatility cap ───────────────────────────────────────────────
    atr_pct = safe_float(row['atr_pct'])    # type: ignore[index]
    if atr_pct > VOLATILITY_MAX_ATR:
        return 'SKIP', f"Volatility too high: ATR%={atr_pct:.3f}>{VOLATILITY_MAX_ATR}"

    # ── L2: Sideway filter ────────────────────────────────────────────────
    adx    = safe_float(row['adx'])         # type: ignore[index]
    slope5 = safe_float(row['ema20_slope_pct'])  # type: ignore[index]

    if adx < ADX_THRESHOLD:
        return 'SKIP', f"Sideway: ADX={adx:.1f}<{ADX_THRESHOLD}"
    if abs(slope5) < SLOPE_THRESHOLD:
        return 'SKIP', f"Sideway: slope5m={slope5:.4f}%<{SLOPE_THRESHOLD}"

    # ── L3: Weighted multi-TF trend ───────────────────────────────────────
    up_w, down_w = _weighted_slope_vote(row)
    total_w = sum(_TF_WEIGHTS.values())          # = 7.0
    min_w   = sum(sorted(_TF_WEIGHTS.values())[-MIN_TREND_FRAMES:])  # top 3 weights
    # Yêu cầu: ít nhất MIN_TREND_FRAMES khung lớn nhất cùng chiều
    slopes_list = [
        safe_float(row['ema20_slope_pct']),
        safe_float(row.get('ema20_15m_slope_pct', 0)),    # type: ignore[attr-defined]
        safe_float(row.get('ema20_30m_slope_pct', 0)),
        safe_float(row.get('ema20_1h_slope_pct', 0)),
    ]
    up_count   = sum(1 for s in slopes_list if s > 0)
    down_count = sum(1 for s in slopes_list if s < 0)

    if max(up_count, down_count) < MIN_TREND_FRAMES:
        return 'SKIP', f"Trend alignment: UP={up_count}/4 DOWN={down_count}/4 < {MIN_TREND_FRAMES}"

    # ADX direction confirmation: DI diff phải agree với slope direction
    di_diff = safe_float(row.get('di_diff', 0))   # type: ignore[attr-defined]
    if up_count >= MIN_TREND_FRAMES and di_diff < -8:
        return 'SKIP', f"ADX DI conflict: slope UP but -DI dominant ({di_diff:.1f})"
    if down_count >= MIN_TREND_FRAMES and di_diff > 8:
        return 'SKIP', f"ADX DI conflict: slope DOWN but +DI dominant ({di_diff:.1f})"

    # ── L4: ML consensus ─────────────────────────────────────────────────
    if ml_prob_up is None or ml_details is None:
        return 'SKIP', "ML not available"

    prob_down    = 1 - ml_prob_up
    votes_ml_up  = sum(1 for p in ml_details.values() if p > 0.5)
    votes_ml_dn  = len(ml_details) - votes_ml_up
    min_votes    = max(2, int(len(ml_details) * CONSENSUS_RATIO))

    if votes_ml_up < min_votes and votes_ml_dn < min_votes:
        return 'SKIP', f"No ML consensus: UP={votes_ml_up} DOWN={votes_ml_dn} (need {min_votes})"

    if ml_prob_up < ML_PROB_THRESHOLD and prob_down < ML_PROB_THRESHOLD:
        return 'SKIP', f"ML prob too low: UP={ml_prob_up:.3f} DOWN={prob_down:.3f}"

    # ── L5: Technical score ───────────────────────────────────────────────
    score_up, score_down = compute_technical_score(row)
    threshold = TECH_SCORE_CAUTIOUS if risk_state.is_cautious else TECH_SCORE_THRESHOLD

    # ── L6: Final entry decision ──────────────────────────────────────────
    direction = None

    if (
        up_count >= MIN_TREND_FRAMES
        and votes_ml_up >= min_votes
        and ml_prob_up >= ML_PROB_THRESHOLD
        and score_up >= threshold
        and score_up > score_down
    ):
        direction = 'UP'

    elif (
        down_count >= MIN_TREND_FRAMES
        and votes_ml_dn >= min_votes
        and prob_down >= ML_PROB_THRESHOLD
        and score_down >= threshold
        and score_down > score_up
    ):
        direction = 'DOWN'

    if direction is None:
        return 'SKIP', (
            f"Weak signal: scoreUP={score_up:.1f} scoreD={score_down:.1f} "
            f"threshold={threshold:.1f} prob={ml_prob_up:.3f}"
        )

    # ── Confidence label ──────────────────────────────────────────────────
    active_score = score_up if direction == 'UP' else score_down
    conf = score_confidence(active_score, threshold)
    # Yêu cầu thêm: HIGH prob để vào lệnh ở cautious mode
    if risk_state.is_cautious and ml_prob_up < ML_HIGH_CONF and direction == 'UP':
        return 'SKIP', f"Cautious mode: need prob>={ML_HIGH_CONF}, got {ml_prob_up:.3f}"
    if risk_state.is_cautious and prob_down < ML_HIGH_CONF and direction == 'DOWN':
        return 'SKIP', f"Cautious mode: need prob>={ML_HIGH_CONF}, got {prob_down:.3f}"

    reason = (
        f"{direction} | trend:{up_count if direction=='UP' else down_count}/4 "
        f"score:{active_score:.1f} prob:{ml_prob_up:.3f} conf:{conf}"
    )
    return direction, reason
