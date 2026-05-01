import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from config_backtest import (
    SYMBOL, INTERVAL, EXTRA_TIMEFRAMES, LOOKBACK_WINDOW,
    STRONG_PROB_THRESHOLD, USE_TREND_FILTER, EMA_TREND_PERIOD,
    MODEL_FILE, CACHE_DIR, LGBM_PARAMS, TEST_DATE_FILE, OUTPUT_BACKTEST
)
from data_loader import get_all_data, load_merged_data, save_merged_data
from indicators import compute_indicators, merge_with_higher_tf
from model_manager import ModelManager

# Múi giờ Việt Nam
VN_TZ = pytz.timezone('Asia/Ho_Chi_Minh')

def read_test_dates():
    if not os.path.exists(TEST_DATE_FILE):
        raise FileNotFoundError(f"File {TEST_DATE_FILE} không tồn tại.")
    encodings = ['utf-8-sig', 'utf-8', 'cp1252', 'latin-1']
    for enc in encodings:
        try:
            with open(TEST_DATE_FILE, 'r', encoding=enc) as f:
                lines = [l.strip() for l in f if l.strip()]
            if len(lines) >= 2:
                print(f"✅ Đọc file {TEST_DATE_FILE} với encoding {enc}")
                return lines[0], lines[1]
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Không thể đọc file {TEST_DATE_FILE}.")

def main():
    print("=" * 60)
    print(" BACKTEST – Dự đoán trên tập test (giờ VN trong testdate.txt)")
    print("=" * 60)

    test_start_str, test_end_str = read_test_dates()
    fmt = "%Y-%m-%d %H:%M"
    try:
        test_start_vn = datetime.strptime(test_start_str, fmt)
        test_end_vn   = datetime.strptime(test_end_str, fmt)
    except ValueError as e:
        print(f"❌ Lỗi định dạng ngày: {e}")
        return

    # Chuyển giờ VN -> UTC
    test_start_utc = VN_TZ.localize(test_start_vn).astimezone(pytz.UTC)
    test_end_utc   = VN_TZ.localize(test_end_vn).astimezone(pytz.UTC)

    if test_end_utc <= test_start_utc:
        print("❌ test_end phải SAU test_start (theo giờ VN).")
        return

    train_start_utc = test_start_utc - timedelta(days=120)
    overall_start = train_start_utc.strftime(fmt)
    overall_end   = test_end_utc.strftime(fmt)

    print(f"\n📥 Tải dữ liệu từ {overall_start} đến {overall_end} (UTC)...")
    intervals_to_fetch = [INTERVAL] + EXTRA_TIMEFRAMES
    all_data = get_all_data(SYMBOL, CACHE_DIR, intervals_to_fetch, overall_start, overall_end)

    df_merged = load_merged_data(SYMBOL, CACHE_DIR, overall_start, overall_end)
    if df_merged is not None:
        print("✅ Đã sử dụng dữ liệu merge từ cache.")
    else:
        print("\n🧮 Tính chỉ báo kỹ thuật...")
        df_5m = all_data[INTERVAL]
        dict_htf = {tf: all_data[tf] for tf in EXTRA_TIMEFRAMES}
        df_5m_ind = compute_indicators(df_5m)
        dict_htf_ind = {}
        for tf, df_htf in dict_htf.items():
            dict_htf_ind[tf] = compute_indicators(df_htf)

        print("🔗 Hợp nhất dữ liệu đa khung vào 5m...")
        df_merged = merge_with_higher_tf(df_5m_ind, dict_htf_ind)
        print(f"   Kích thước DataFrame sau merge: {df_merged.shape}")
        if df_merged.empty:
            print("❌ Dữ liệu sau merge rỗng.")
            return
        save_merged_data(df_merged, SYMBOL, CACHE_DIR, overall_start, overall_end)

    model_mgr = ModelManager(MODEL_FILE, LGBM_PARAMS, LOOKBACK_WINDOW)
    X, y, timestamps = model_mgr.prepare_data(df_merged)
    print(f"Tổng mẫu: {len(X)}, Số đặc trưng mỗi mẫu: {X.shape[1]}")

    # Lọc mẫu test theo UTC
    ts_utc = pd.DatetimeIndex(timestamps).tz_localize(pytz.UTC)
    test_mask = (ts_utc >= test_start_utc) & (ts_utc <= test_end_utc)
    X_test, y_test = X[test_mask], y[test_mask]
    ts_test_utc = timestamps[test_mask]

    if len(X_test) == 0:
        print("❌ Không có mẫu test nào. Kiểm lại múi giờ hoặc dữ liệu cache.")
        return

    if not model_mgr.load_model(current_feature_count=X.shape[1]):
        print("❌ Không thể nạp mô hình. Hãy chạy train.py trước.")
        return

    print("\n🔮 Đang dự đoán...")
    proba_up = model_mgr.predict_proba(X_test)

    use_filter = USE_TREND_FILTER
    ema_col = f'ema{EMA_TREND_PERIOD}'
    if use_filter and ema_col not in df_merged.columns:
        print(f"⚠️ Cột {ema_col} không tồn tại, tắt bộ lọc xu hướng.")
        use_filter = False

    results = []
    for i in range(len(X_test)):
        ts_utc = pd.Timestamp(ts_test_utc[i])
        if ts_utc in df_merged.index:
            row = df_merged.loc[ts_utc]
            open_p = row['open']
            close_p = row['close']
        else:
            open_p = close_p = None
            row = None

        prob = proba_up[i]
        signal = "SKIP"
        reason = ""

        if prob >= STRONG_PROB_THRESHOLD:
            if use_filter and row is not None:
                if close_p > row[ema_col]:
                    signal = "UP"
                    reason = f"UP: prob={prob:.3f} >= {STRONG_PROB_THRESHOLD}, price>{ema_col}={row[ema_col]:.2f}"
                else:
                    reason = f"SKIP: UP prob={prob:.3f} nhưng price <= {ema_col} ({row[ema_col]:.2f})"
            else:
                signal = "UP"
                reason = f"UP: prob={prob:.3f} >= {STRONG_PROB_THRESHOLD}"
        elif prob <= 1 - STRONG_PROB_THRESHOLD:
            if use_filter and row is not None:
                if close_p < row[ema_col]:
                    signal = "DOWN"
                    reason = f"DOWN: prob_down={1-prob:.3f} >= {STRONG_PROB_THRESHOLD}, price<{ema_col}={row[ema_col]:.2f}"
                else:
                    reason = f"SKIP: DOWN prob={1-prob:.3f} nhưng price >= {ema_col} ({row[ema_col]:.2f})"
            else:
                signal = "DOWN"
                reason = f"DOWN: prob_down={1-prob:.3f} >= {STRONG_PROB_THRESHOLD}"
        else:
            reason = f"SKIP: UP prob={prob:.3f} trung tính (ngưỡng {STRONG_PROB_THRESHOLD})"

        actual = y_test[i]
        if signal == "SKIP":
            winlose = "SKIP"
        else:
            pred_class = 1 if signal == "UP" else 0
            winlose = "WIN" if pred_class == actual else "LOSE"

        # Chuyển timestamp về giờ VN để hiển thị
        ts_vn = ts_utc.tz_localize(pytz.UTC).astimezone(VN_TZ).replace(tzinfo=None)

        results.append({
            "timestamp": ts_vn,
            "open": open_p,
            "close": close_p,
            "signal": signal,
            "winlose": winlose,
            "prob_up": round(prob, 4),
            "reason": reason
        })

    df_res = pd.DataFrame(results)
    # Sắp xếp cột theo thứ tự mong muốn
    df_res = df_res[["timestamp", "open", "close", "signal", "winlose", "prob_up", "reason"]]
    df_res.to_excel(OUTPUT_BACKTEST, index=False)
    print(f"✅ Đã xuất kết quả backtest: {OUTPUT_BACKTEST}")

    # Thống kê
    total = len(df_res)
    count_up = (df_res['signal'] == 'UP').sum()
    count_down = (df_res['signal'] == 'DOWN').sum()
    count_skip = (df_res['signal'] == 'SKIP').sum()

    print("\n📊 THỐNG KÊ BACKTEST:")
    print(f"   Tổng số nến test: {total}")
    print(f"   UP   : {count_up} ({100*count_up/total:.1f}%)")
    print(f"   DOWN : {count_down} ({100*count_down/total:.1f}%)")
    print(f"   SKIP : {count_skip} ({100*count_skip/total:.1f}%)")

    up_trades = df_res[df_res['signal'] == 'UP']
    down_trades = df_res[df_res['signal'] == 'DOWN']
    up_wins = (up_trades['winlose'] == 'WIN').sum()
    up_losses = (up_trades['winlose'] == 'LOSE').sum()
    down_wins = (down_trades['winlose'] == 'WIN').sum()
    down_losses = (down_trades['winlose'] == 'LOSE').sum()

    if (up_wins+up_losses) > 0:
        print(f"   UP   -> Thắng: {up_wins}, Thua: {up_losses}, WR: {100*up_wins/(up_wins+up_losses):.1f}%")
    else:
        print("   UP   -> Không có lệnh")
    if (down_wins+down_losses) > 0:
        print(f"   DOWN -> Thắng: {down_wins}, Thua: {down_losses}, WR: {100*down_wins/(down_wins+down_losses):.1f}%")
    else:
        print("   DOWN -> Không có lệnh")

    all_trades = df_res[df_res['signal'] != 'SKIP']
    if len(all_trades) > 0:
        total_wins = (all_trades['winlose'] == 'WIN').sum()
        total_losses = (all_trades['winlose'] == 'LOSE').sum()
        days = (test_end_vn - test_start_vn).days + 1
        avg_trades = len(all_trades) / days if days > 0 else 0
        print(f"\n   Tổng lệnh vào: {len(all_trades)} | Thắng: {total_wins} | Thua: {total_losses}")
        print(f"   Win Rate tổng: {100*total_wins/len(all_trades):.2f}%")
        print(f"   Số lệnh trung bình/ngày: {avg_trades:.1f}")
    else:
        print("\n   ⚠️ Không có lệnh nào được vào.")

if __name__ == "__main__":
    main()