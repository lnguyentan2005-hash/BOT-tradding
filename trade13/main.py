# === FILE: main.py ===
"""
main.py – Bot giao dịch realtime BTC/USDT 5m.
v5.0: Timing chính xác (fetch tại xx:xx:55), display chuyên nghiệp,
      Telegram notifications, session-aware, pending trade tracking.
Chạy: py main.py
"""
import os, sys, time
from datetime import datetime, timedelta

import pytz
import pandas as pd
from colorama import init, Fore, Style

from config import (
    LSTM_MODEL_PATH, LGBM_MODEL_PATH, XGB_MODEL_PATH,
    SCALER_PATH, FEATURE_COLS_PATH,
    TIMEFRAME_PRIMARY, TIMEFRAME_15M, TIMEFRAME_30M, TIMEFRAME_1H,
    LOOKBACK_LIVE, TIMEZONE, TECH_SCORE_THRESHOLD, TECH_SCORE_CAUTIOUS,
    ML_HIGH_CONF,
)
from data_layer import DataLayer
from features import generate_features
from ml_model import EnsembleModel
from decision_engine import make_decision, RiskState
from scoring import compute_technical_score
from logger import init_log, log_trade
from utils import (
    now_vn, now_utc, slope_icon, pct_str, price_str,
    session_name, send_telegram, format_signal_telegram,
)

init(autoreset=True)
_tz  = pytz.timezone(TIMEZONE)
_utc = pytz.utc

VERSION = "v5.0"


class TradingBot:
    """Bot giao dịch realtime BTC/USDT 5m — pipeline đa khung."""

    def __init__(self) -> None:
        # Components
        self.data  = DataLayer()
        self.model = EnsembleModel()
        self.risk  = RiskState()

        # State
        self.last_signal : str | None   = None
        self.last_close  : float | None = None
        self.stats       = {'wins': 0, 'losses': 0, 'skips': 0}

        # Pending trade: lưu tín hiệu hiện tại, chờ nến sau để xác định kết quả
        self.pending: dict | None = None

        # Boot: kiểm tra model
        self._require_models()
        self.model.load_models()

    # ── Startup checks ────────────────────────────────────────────────────

    def _require_models(self) -> None:
        required = [LSTM_MODEL_PATH, LGBM_MODEL_PATH, XGB_MODEL_PATH, SCALER_PATH, FEATURE_COLS_PATH]
        missing  = [p for p in required if not os.path.exists(p)]
        if missing:
            print(f"{Fore.RED}[ERROR] Model chưa được huấn luyện.")
            print("➜ Hãy chạy: py train.py trước khi chạy bot.")
            sys.exit(1)

    # ── Main loop ─────────────────────────────────────────────────────────

    def run(self) -> None:
        init_log()
        self._print_header()

        while True:
            # Phân tích ngay khi khởi động
            try:
                self._cycle()
            except Exception as exc:
                print(f"{Fore.RED}[ERROR] {exc}{Style.RESET_ALL}")

            # Tính thời gian chờ đến xx:xx:55 của nến tiếp theo
            wait = self._seconds_to_next_fetch()
            self._wait_countdown(wait)

    def _cycle(self) -> None:
        """Một chu kỳ phân tích: fetch → feature → ML → decide → log."""
        t0 = time.perf_counter()
        fetch_time = now_vn().strftime('%H:%M:%S.%f')[:12]

        # Fetch data
        df_5m  = self.data.fetch_ohlcv(TIMEFRAME_PRIMARY, LOOKBACK_LIVE)
        df_15m = self.data.fetch_ohlcv(TIMEFRAME_15M, 300)
        df_30m = self.data.fetch_ohlcv(TIMEFRAME_30M, 300)
        df_1h  = self.data.fetch_ohlcv(TIMEFRAME_1H,  300)

        fetch_done = time.perf_counter()
        fetch_ms   = (fetch_done - t0) * 1000

        if df_5m.empty:
            print(f"{Fore.YELLOW}  ⚠ No data received from exchange.{Style.RESET_ALL}")
            return

        # Settle previous trade result (dùng nến mới nhất)
        self._settle_pending(df_5m)

        # Generate features
        df_feat, _ = generate_features(df_5m, df_15m, df_30m, df_1h, dropna=False)
        if df_feat.empty:
            return

        latest = df_feat.iloc[-1]

        # ML prediction
        prob_up, details = self.model.predict_proba(df_feat)

        # Decision
        signal, reason = make_decision(latest, prob_up, details, self.risk)

        # Display
        elapsed = (time.perf_counter() - t0) * 1000
        self._print_dashboard(latest, prob_up, details, signal, reason, fetch_time, elapsed)

        # Store pending trade
        if signal in ('UP', 'DOWN'):
            self.pending = {
                'signal' : signal,
                'close'  : latest['close'],
                'reason' : reason,
                'row'    : latest,
                'prob_up': prob_up,
                'details': details,
            }
            if TELEGRAM_NOTIFY:
                send_telegram(format_signal_telegram(
                    signal, latest['close'], prob_up or 0,
                    *compute_technical_score(latest),
                    reason, now_vn().strftime('%H:%M:%S'),
                ))
        else:
            self.stats['skips'] += 1

    def _settle_pending(self, df_5m: pd.DataFrame) -> None:
        """Xác định WIN/LOSS cho lệnh đang pending, ghi log."""
        if self.pending is None:
            return
        if len(df_5m) < 2:
            return
        cur_close  = df_5m.iloc[-1]['close']
        prev_close = self.pending['close']
        signal     = self.pending['signal']

        won    = (signal == 'UP' and cur_close > prev_close) or \
                 (signal == 'DOWN' and cur_close < prev_close)
        result = 'WIN' if won else 'LOSS'

        self.risk.record_result(won)
        if won:
            self.stats['wins']   += 1
        else:
            self.stats['losses'] += 1

        log_trade(
            row        = self.pending['row'],
            ml_prob    = self.pending['prob_up'],
            ml_details = self.pending['details'],
            decision   = signal,
            reason     = self.pending['reason'],
            result     = result,
        )

        color = Fore.GREEN if won else Fore.RED
        icon  = '✓ WIN' if won else '✗ LOSS'
        print(f"\n{color}  ► Prev {signal}: {prev_close:,.2f} → {cur_close:,.2f} │ {icon}{Style.RESET_ALL}")

        self.last_signal = signal
        self.last_close  = prev_close
        self.pending     = None

    # ── Display ───────────────────────────────────────────────────────────

    def _print_header(self) -> None:
        print(Fore.CYAN + "╔══════════════════════════════════════════════════════════════╗")
        print(f"║  BTC Promax AI {VERSION}  │  BTC/USDT 5m  │  Starting...         ║")
        print("╚══════════════════════════════════════════════════════════════╝" + Style.RESET_ALL)

    def _print_dashboard(
        self, row, prob_up, details, signal, reason, fetch_time, elapsed_ms
    ) -> None:
        vn_now  = now_vn().strftime('%H:%M:%S')
        utc_now = now_utc().strftime('%H:%M:%S')
        sess    = session_name()

        close   = row['close']
        delta   = row.get('returns_1', 0) or 0
        slope5  = row.get('ema20_slope_pct',    0) or 0
        slope15 = row.get('ema20_15m_slope_pct', 0) or 0
        slope30 = row.get('ema20_30m_slope_pct', 0) or 0
        slope1h = row.get('ema20_1h_slope_pct',  0) or 0
        rsi     = row.get('rsi',   50) or 50
        adx     = row.get('adx',    0) or 0
        atr_pct = row.get('atr_pct', 0) or 0
        vol_r   = row.get('volume_ratio', 1) or 1
        macd_d  = row.get('macd_diff', 0) or 0
        stoch_k = row.get('stoch_k', 50) or 50
        stoch_d = row.get('stoch_d', 50) or 50
        bb_pct  = row.get('bb_pct', 0.5) or 0.5
        vwap_d  = row.get('vwap_dist', 0) or 0

        score_up, score_down = compute_technical_score(row)
        threshold = TECH_SCORE_CAUTIOUS if self.risk.is_cautious else TECH_SCORE_THRESHOLD

        slopes_list = [slope5, slope15, slope30, slope1h]
        up_votes   = sum(1 for s in slopes_list if s > 0)
        down_votes = sum(1 for s in slopes_list if s < 0)

        total_trades = self.stats['wins'] + self.stats['losses']
        wr = self.stats['wins'] / total_trades * 100 if total_trades else 0.0

        # Prev trade display
        prev_str = '—'
        if self.last_signal and self.last_close:
            if self.last_signal == 'UP':
                res = (Fore.GREEN + '✓ WIN') if close > self.last_close else (Fore.RED + '✗ LOSS')
            else:
                res = (Fore.GREEN + '✓ WIN') if close < self.last_close else (Fore.RED + '✗ LOSS')
            prev_str = f"{self.last_signal} {self.last_close:,.2f}→{close:,.2f} │ {res}{Style.RESET_ALL}"

        # Signal color
        if signal == 'UP':
            sig_color = Fore.GREEN
            sig_block = '██ UP ██  '
        elif signal == 'DOWN':
            sig_color = Fore.RED
            sig_block = '██ DOWN ██'
        else:
            sig_color = Fore.YELLOW
            sig_block = '── SKIP ──'

        # Confidence
        conf = 'HIGH' if prob_up and (prob_up > ML_HIGH_CONF or (1-prob_up) > ML_HIGH_CONF) else 'MEDIUM'
        mode = 'Cautious⚠' if self.risk.is_cautious else 'Normal   '

        W = 62  # box width

        def row_line(label: str, content: str) -> str:
            inner = f"  {label:<10}│  {content}"
            return f"║{inner:<{W}}║"

        print("\n" + "╔" + "═"*W + "╗")
        print(f"║  BTC Promax AI {VERSION}  │  {vn_now} (VN) {utc_now} (UTC){'':>3}║")
        print(f"║  Session: {sess:<12}│  Fetch: {fetch_time}  ({elapsed_ms:.0f}ms){'':>6}║")
        print("╠" + "═"*W + "╣")
        print(row_line("PRICE", f"Close: {price_str(close)}  Δ: {pct_str(delta*100, 2)}"))
        print(row_line("TREND", f"5m:{slope_icon(slope5)}{slope5:+.3f}% 15m:{slope_icon(slope15)}{slope15:+.3f}% 30m:{slope_icon(slope30)} 1h:{slope_icon(slope1h)}"))
        print(row_line("MOMENTUM", f"RSI:{rsi:.1f}  ADX:{adx:.1f}  ATR%:{atr_pct:.3f}  Vol:{vol_r:.2f}x"))
        print(row_line("MACD", f"diff:{macd_d:.1f}  StochK:{stoch_k:.1f} D:{stoch_d:.1f}  BB%:{bb_pct:.2f}"))
        print(row_line("VWAP", f"dist:{vwap_d:+.3f}%"))
        print("╠" + "═"*W + "╣")
        if details:
            avg_str = f"{prob_up:.3f}" if prob_up else 'N/A'
            print(row_line("ML SCORE", f"LSTM:{details['lstm']:.3f}  LGB:{details['lgbm']:.3f}  XGB:{details['xgb']:.3f}  Avg:{avg_str}"))
        else:
            print(row_line("ML SCORE", "Not available"))
        print(row_line("TA SCORE", f"UP:{score_up:.1f}  DOWN:{score_down:.1f}  Threshold:{threshold:.1f}"))
        print(row_line("TREND", f"UP_votes:{up_votes}/4  DOWN_votes:{down_votes}/4"))
        print("╠" + "═"*W + "╣")
        print(f"║  ➤ SIGNAL │  {sig_color}{sig_block}{Style.RESET_ALL}  │  Confidence: {conf}{'':>20}║")
        reason_short = reason[:48]
        print(row_line("REASON", f"{reason_short}"))
        print("╠" + "═"*W + "╣")
        print(row_line("PREV", prev_str[:W-16]))
        print(row_line("SESSION", f"W:{self.stats['wins']}  L:{self.stats['losses']}  Skip:{self.stats['skips']}  WR:{wr:.1f}%"))
        print(row_line("RISK", f"{mode}  │  Streak: {self.risk.consecutive_losses}  │  Next: {self._secs_to_candle()}s"))
        print("╚" + "═"*W + "╝")

    # ── Timing helpers ────────────────────────────────────────────────────

    def _next_candle_close_utc(self) -> datetime:
        now  = now_utc()
        mins = now.minute
        next5 = ((mins // 5) + 1) * 5
        base  = now.replace(minute=0, second=0, microsecond=0)
        return base + timedelta(minutes=next5)

    def _secs_to_candle(self) -> int:
        return max(0, int((self._next_candle_close_utc() - now_utc()).total_seconds()))

    def _seconds_to_next_fetch(self) -> float:
        """Chờ đến xx:xx:55 (5s trước khi nến đóng) để fetch data sớm."""
        target = self._next_candle_close_utc() - timedelta(seconds=5)
        delta  = (target - now_utc()).total_seconds()
        return max(1.0, delta)

    def _wait_countdown(self, seconds: float) -> None:
        """Hiển thị countdown đến lần phân tích tiếp theo."""
        for remaining in range(int(seconds), 0, -1):
            print(f"  ⏳ Next analysis in {remaining:3d}s ...  ", end='\r')
            time.sleep(1)
        print(" " * 40, end='\r')


# ── Telegram toggle (sửa nếu muốn bật) ───────────────────────────────────────
TELEGRAM_NOTIFY = False   # True để bật Telegram


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    bot = TradingBot()
    try:
        bot.run()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}  Bot stopped by user.{Style.RESET_ALL}")
