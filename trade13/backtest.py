# === FILE: backtest.py ===
"""
backtest.py – Backtest thuật toán trên khoảng thời gian chỉ định.
v5.0.1: Batch ML inference – giảm thời gian chạy 10-20x.
"""
import os, sys, time
import pandas as pd
import numpy as np
from tqdm import tqdm
from numpy.lib.stride_tricks import sliding_window_view
import pytz

# ======================================================
TEST_START  = '2026-04-01'   # <-- SỬA NGÀY BẮT ĐẦU TEST
TEST_END    = '2026-04-29'   # <-- SỬA NGÀY KẾT THÚC TEST
# ======================================================

from config import (
    LSTM_MODEL_PATH, LGBM_MODEL_PATH, XGB_MODEL_PATH,
    SCALER_PATH, FEATURE_COLS_PATH,
    TIMEFRAME_PRIMARY, TIMEFRAME_15M, TIMEFRAME_30M, TIMEFRAME_1H,
    TIMEZONE, BACKTEST_LOG_CSV,
)
from data_layer import DataLayer
from features import generate_features
from ml_model import EnsembleModel
from decision_engine import make_decision, RiskState
from logger import init_backtest_log, log_backtest_trade

_tz = pytz.timezone(TIMEZONE)

_PROFIT_WIN  =  0.95
_PROFIT_LOSS = -1.00


def _check_models() -> None:
    required = [LSTM_MODEL_PATH, LGBM_MODEL_PATH, XGB_MODEL_PATH, SCALER_PATH, FEATURE_COLS_PATH]
    missing  = [p for p in required if not os.path.exists(p)]
    if missing:
        print(f"\n[ERROR] Model files missing:")
        for p in missing:
            print(f"  ✗ {p}")
        print("\n➜ Hãy chạy: py train.py trước khi backtest.\n")
        sys.exit(1)


def _load_data(since: pd.Timestamp, until: pd.Timestamp) -> tuple:
    dl = DataLayer()
    print(f"\n  Downloading test data: {since.date()} → {until.date()}")
    t0 = time.time()
    df_5m  = dl.fetch_ohlcv_range(TIMEFRAME_PRIMARY, since, until, show_progress=True)
    df_15m = dl.fetch_ohlcv_range(TIMEFRAME_15M,     since, until, show_progress=True)
    df_30m = dl.fetch_ohlcv_range(TIMEFRAME_30M,     since, until, show_progress=True)
    df_1h  = dl.fetch_ohlcv_range(TIMEFRAME_1H,      since, until, show_progress=True)
    print(f"  5m={len(df_5m):,}  15m={len(df_15m):,}  30m={len(df_30m):,}  1h={len(df_1h):,}  "
          f"({time.time()-t0:.1f}s)")
    return df_5m, df_15m, df_30m, df_1h


def run_backtest(test_start: str, test_end: str) -> None:
    since = pd.Timestamp(test_start, tz='UTC')
    until = pd.Timestamp(test_end,   tz='UTC')

    print("╔══════════════════════════════════════════════════╗")
    print("║   BTC Promax AI v5.0  │  Backtesting             ║")
    print(f"║   Period: {test_start} → {test_end}       ║")
    print("╚══════════════════════════════════════════════════╝")

    _check_models()
    df_5m, df_15m, df_30m, df_1h = _load_data(since, until)

    t_feat = time.time()
    print("\n  Generating features...")
    df_feat, feature_cols = generate_features(df_5m, df_15m, df_30m, df_1h, dropna=True)
    if df_feat.empty:
        print("[ERROR] No features generated. Check data download.")
        return
    print(f"  Feature matrix: {df_feat.shape[0]:,} rows × {len(feature_cols)} cols  "
          f"({time.time()-t_feat:.1f}s)")

    model = EnsembleModel()
    model.load_models()
    risk  = RiskState()
    init_backtest_log()

    N = len(df_feat)
    lookback = model.lookback
    total_candles = N - 1                     # số nến có thể kiểm tra
    if N <= lookback + 1:
        print("[ERROR] Not enough data after lookback.")
        return

    # ── Chuẩn bị scaled data & batch predict ─────────────────────────────
    t_batch = time.time()
    data_scaled = model.scaler.transform(df_feat[feature_cols].values)

    # Tạo sliding windows cho LSTM (chỉ lấy cửa sổ kết thúc tại i ≥ lookback)
    # sliding_window_view(scaled, lookback, axis=0) → shape (N-lookback+1, lookback, F)
    X_seq_all = sliding_window_view(data_scaled, lookback, axis=0)[1:]   # bỏ cửa sổ i<lookback
    # Chuyển đúng shape cho LSTM: (samples, timesteps, features)
    X_seq_all = X_seq_all.transpose(0, 2, 1)
    X_flat_all = data_scaled[lookback:N]         # dòng tương ứng i = lookback → N-1

    print(f"  Batch arrays: X_seq {X_seq_all.shape}  X_flat {X_flat_all.shape}")

    # Predict toàn bộ
    lstm_probs = model.lstm.predict(X_seq_all, batch_size=1024, verbose=0).flatten()
    lgbm_probs = model.lgbm.predict_proba(X_flat_all)[:, 1]
    xgb_probs  = model.xgb.predict_proba(X_flat_all)[:, 1]
    avg_probs  = (lstm_probs + lgbm_probs + xgb_probs) / 3.0
    print(f"  Batch inference done in {time.time()-t_batch:.1f}s")

    # ── Main loop (chỉ logic, không inference) ───────────────────────────
    results     : list[dict] = []
    pending_signal : str | None = None
    pending_close  : float      = 0.0
    pending_row    = None
    pending_prob   : float | None = None
    pending_detail : dict | None  = None
    pending_reason : str          = ''

    wins = losses = skips = 0

    pbar = tqdm(range(lookback, N), desc='Backtesting', unit='candle', ncols=80)
    t_loop = time.time()

    for i in pbar:
        idx = i - lookback            # vị trí trong batch arrays
        row = df_feat.iloc[i]

        # ── Settle previous signal ────────────────────────────────────────
        if pending_signal is not None:
            cur_close = row['close']
            won = (pending_signal == 'UP' and cur_close > pending_close) or \
                  (pending_signal == 'DOWN' and cur_close < pending_close)
            profit = _PROFIT_WIN if won else _PROFIT_LOSS
            result = 'WIN' if won else 'LOSS'
            risk.record_result(won)

            if won:
                wins += 1
            else:
                losses += 1

            results.append({
                'timestamp_vn': pending_row['timestamp'].tz_convert(_tz),
                'signal'  : pending_signal,
                'close_in': pending_close,
                'close_out': cur_close,
                'won'     : won,
                'profit'  : profit,
                'reason'  : pending_reason,
            })

            log_backtest_trade(
                row     = pending_row,
                ml_prob = pending_prob,
                ml_details = pending_detail,
                decision= pending_signal,
                reason  = pending_reason,
                result  = result,
            )

            pending_signal = None
            total_so_far = wins + losses
            wr = wins / total_so_far * 100 if total_so_far else 0
            pbar.set_postfix_str(f'W:{wins} L:{losses} WR:{wr:.1f}%')

        # ── Current candle decision ───────────────────────────────────────
        risk.tick()
        prob_up = avg_probs[idx]
        details = {'lstm': lstm_probs[idx], 'lgbm': lgbm_probs[idx], 'xgb': xgb_probs[idx]}
        signal, reason = make_decision(row, prob_up, details, risk)

        if signal in ('UP', 'DOWN'):
            pending_signal = signal
            pending_close  = row['close']
            pending_row    = row
            pending_prob   = prob_up
            pending_detail = details
            pending_reason = reason
        else:
            skips += 1

    pbar.close()
    print(f"  Loop time: {time.time()-t_loop:.1f}s")

    # ── Summary ───────────────────────────────────────────────────────────
    if not results:
        print("\n  No trades executed in this period.")
        return

    df_res   = pd.DataFrame(results)
    # ✅ Xoá timezone để xuất Excel không bị lỗi
    if 'timestamp_vn' in df_res.columns:
        df_res['timestamp_vn'] = df_res['timestamp_vn'].dt.tz_localize(None)

    total    = len(df_res)
    wr_final = wins / total * 100 if total else 0
    net      = df_res['profit'].sum()
    gross_w  = df_res[df_res['won']]['profit'].sum()
    gross_l  = -df_res[~df_res['won']]['profit'].sum()
    pf       = gross_w / gross_l if gross_l else float('inf')

    max_cl = streak = 0
    for w in df_res['won']:
        streak = 0 if w else streak + 1
        max_cl = max(max_cl, streak)

    ev = net / total

    print(f"\n{'═'*50}")
    print(f"╔══════════════ BACKTEST RESULTS ══════════════╗")
    print(f"║  Period    : {test_start} → {test_end}   ║")
    print(f"║  Candles   : {total_candles:,} analyzed              ║")
    print(f"╠══════════════════════════════════════════════╣")
    print(f"║  Signals   : {total} ({total/total_candles*100:.1f}%)  │  Skip: {skips}     ║")
    print(f"║  WIN       : {wins} ({wr_final:.1f}%)  │  LOSS: {losses}     ║")
    print(f"║  Winrate   : {wr_final:.1f}%                          ║")
    print(f"║  Net Profit: {net:+.2f} units                    ║")
    print(f"║  Prof.Factor: {pf:.2f}                          ║")
    print(f"║  EV/trade  : {ev:+.3f} units                    ║")
    print(f"║  Max ConsL : -{max_cl} consecutive losses          ║")
    print(f"╚══════════════════════════════════════════════╝")

    if wr_final >= 75:
        print("  🎯 Winrate TARGET ≥75% ACHIEVED!")
    else:
        print(f"  ⚠️  Winrate {wr_final:.1f}% below target 75%")

    out_xlsx = f"backtest_{test_start}_{test_end}.xlsx"
    df_res.to_excel(out_xlsx, index=False)
    print(f"\n  📄 Results saved: {out_xlsx}")
    print(f"  📄 Log CSV: {BACKTEST_LOG_CSV}\n")


if __name__ == '__main__':
    run_backtest(TEST_START, TEST_END)