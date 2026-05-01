import os
import pandas as pd
import numpy as np
from datetime import datetime
from config_train import (
    SYMBOL, INTERVAL, EXTRA_TIMEFRAMES, LOOKBACK_WINDOW,
    DATES_FILE, CACHE_DIR, MODEL_FILE, LGBM_PARAMS
)
from data_loader import get_all_data, load_merged_data, save_merged_data
from indicators import compute_indicators, merge_with_higher_tf
from model_manager import ModelManager

def read_train_dates():
    if not os.path.exists(DATES_FILE):
        raise FileNotFoundError(f"File {DATES_FILE} không tồn tại.")
    encodings = ['utf-8-sig', 'utf-8', 'cp1252', 'latin-1']
    for enc in encodings:
        try:
            with open(DATES_FILE, 'r', encoding=enc) as f:
                lines = [l.strip() for l in f if l.strip()]
            if len(lines) >= 2:
                print(f"✅ Đọc file {DATES_FILE} với encoding {enc}")
                return lines[0], lines[1]
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Không thể đọc file {DATES_FILE}.")

def main():
    print("=" * 60)
    print(" TRAIN ONLY – Huấn luyện mô hình (không test)")
    print("=" * 60)

    train_start_str, train_end_str = read_train_dates()
    fmt = "%Y-%m-%d %H:%M"
    try:
        train_start = datetime.strptime(train_start_str, fmt)
        train_end   = datetime.strptime(train_end_str, fmt)
    except ValueError as e:
        print(f"❌ Lỗi định dạng ngày: {e}")
        return

    if train_end <= train_start:
        print("❌ train_end phải SAU train_start.")
        return

    overall_start = train_start.strftime(fmt)
    overall_end   = train_end.strftime(fmt)

    print("\n📥 Tải dữ liệu cho toàn bộ khoảng train...")
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
        if df_merged.empty:
            print("❌ Dữ liệu sau merge rỗng.")
            return
        save_merged_data(df_merged, SYMBOL, CACHE_DIR, overall_start, overall_end)

    model_mgr = ModelManager(MODEL_FILE, LGBM_PARAMS, LOOKBACK_WINDOW)
    X, y, timestamps = model_mgr.prepare_data(df_merged)
    print(f"Tổng mẫu: {len(X)}, Số đặc trưng mỗi mẫu: {X.shape[1]}")

    train_mask = (timestamps >= np.datetime64(train_start)) & (timestamps <= np.datetime64(train_end))
    X_train, y_train = X[train_mask], y[train_mask]
    print(f"Mẫu sau khi lọc đúng khoảng train: {len(X_train)}")

    if len(X_train) < 200:
        print(f"❌ Không đủ dữ liệu để train (chỉ có {len(X_train)} mẫu). Cần ít nhất 200.")
        return

    print("\n🤖 Bắt đầu huấn luyện LightGBM...")
    split = int(len(X_train) * 0.8)
    X_tr, X_val = X_train[:split], X_train[split:]
    y_tr, y_val = y_train[:split], y_train[split:]

    model_mgr.train(X_tr, y_tr, X_valid=X_val, y_valid=y_val)
    model_mgr.save_model()
    print("✅ Huấn luyện hoàn tất. Mô hình đã được lưu.")

    if len(X_val) > 0:
        from sklearn.metrics import accuracy_score
        y_pred_val = model_mgr.model.predict(X_val)
        acc_val = accuracy_score(y_val, y_pred_val)
        y_pred_train = model_mgr.model.predict(X_tr)
        acc_train = accuracy_score(y_tr, y_pred_train)
        print(f"📊 Accuracy train: {acc_train:.4f}  |  Accuracy validation: {acc_val:.4f}")
        if acc_train - acc_val > 0.05:
            print("⚠️ Có dấu hiệu overfitting (chênh lệch >5%). Cân nhắc giảm độ phức tạp mô hình.")
        else:
            print("✅ Mô hình không bị overfitting đáng kể.")

if __name__ == "__main__":
    main()