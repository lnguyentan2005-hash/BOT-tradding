# === FILE: train.py ===
"""
train.py – Huấn luyện Ensemble (LSTM + LightGBM + XGBoost) từ dữ liệu lịch sử.
v5.0: Fix evaluation bug, thêm test-set report, feature importance.
Chạy: py train.py
"""
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, classification_report,
)

from config import (
    TRAIN_DAYS, TIMEFRAME_PRIMARY, TIMEFRAME_15M, TIMEFRAME_30M, TIMEFRAME_1H,
    MIN_CANDLES_FOR_TRAIN, LSTM_LOOKBACK,
)
from data_layer import DataLayer
from features import generate_features
from ml_model import EnsembleModel


def download_data(dl: DataLayer, days: int) -> tuple:
    """Tải dữ liệu đa khung từ Binance (có cache)."""
    print(f"\n{'─'*55}")
    print(f"  Downloading {days}-day history (4 timeframes)...")
    print(f"{'─'*55}")
    df_5m  = dl.fetch_history(TIMEFRAME_PRIMARY, days=days)
    df_15m = dl.fetch_history(TIMEFRAME_15M, days=days)
    df_30m = dl.fetch_history(TIMEFRAME_30M, days=days)
    df_1h  = dl.fetch_history(TIMEFRAME_1H, days=days)
    print(f"\n  Loaded: 5m={len(df_5m):,}  15m={len(df_15m):,}  "
          f"30m={len(df_30m):,}  1h={len(df_1h):,} candles")
    return df_5m, df_15m, df_30m, df_1h


def validate_data(df: pd.DataFrame, name: str = '5m') -> None:
    if df.empty:
        print(f"[ERROR] No {name} data downloaded. Check network / exchange.")
        sys.exit(1)
    if len(df) < MIN_CANDLES_FOR_TRAIN:
        print(f"[ERROR] Not enough {name} candles ({len(df)} < {MIN_CANDLES_FOR_TRAIN}).")
        sys.exit(1)
    nulls = df.isnull().sum().sum()
    if nulls > 0:
        print(f"  [{name}] WARNING: {nulls} null values found, will be dropped.")


def evaluate_test_set(
    model: EnsembleModel,
    test_df: pd.DataFrame,
) -> dict:
    """
    Đánh giá model trên test set.
    FIX: Dùng predict_proba cho từng timestamp, không phải scalar.
    """
    y_true, y_pred_proba = [], []
    lookback = model.lookback

    # Cần đủ lookback rows
    if len(test_df) <= lookback + 1:
        print("  [Evaluation] Not enough test data.")
        return {}

    # Build full scaled array
    try:
        data   = test_df[model.feature_cols].values
        scaled = model.scaler.transform(data)
    except Exception as exc:
        print(f"  [Evaluation] Scaling error: {exc}")
        return {}

    # Build targets
    y_all = (test_df['close'].shift(-1) > test_df['close']).astype(int).values
    # Loại dòng cuối (NaN)
    y_all = y_all[:-1]
    n     = len(y_all)

    # LSTM batch predict (hiệu quả hơn vòng lặp)
    import numpy as np
    Xs = np.array([scaled[i:i+lookback] for i in range(n - lookback)])
    if len(Xs) == 0:
        return {}
    lstm_probs = model.lstm.predict(Xs, batch_size=256, verbose=0).flatten()

    X_flat = scaled[lookback:n]
    lgbm_probs = model.lgbm.predict_proba(X_flat)[:, 1]
    xgb_probs  = model.xgb.predict_proba(X_flat)[:, 1]

    avg_probs = (lstm_probs + lgbm_probs + xgb_probs) / 3
    y_true    = y_all[lookback:]

    if len(y_true) == 0:
        return {}

    # Dùng optimal threshold đã tìm được
    threshold = model.opt_threshold
    y_pred    = (avg_probs >= threshold).astype(int)

    auc  = roc_auc_score(y_true, avg_probs)
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    return {
        'n_test' : len(y_true),
        'auc'    : auc,
        'acc'    : acc,
        'prec'   : prec,
        'recall' : rec,
        'f1'     : f1,
        'thresh' : threshold,
        'n_pred_up'  : int(y_pred.sum()),
        'n_pred_down': int((1 - y_pred).sum()),
    }


def print_metrics(m: dict) -> None:
    if not m:
        return
    print(f"\n{'═'*50}")
    print(f"  TEST SET EVALUATION")
    print(f"{'─'*50}")
    print(f"  Samples    : {m['n_test']:,}")
    print(f"  Threshold  : {m['thresh']:.3f}")
    print(f"  AUC-ROC    : {m['auc']:.4f}")
    print(f"  Accuracy   : {m['acc']:.4f}  ({m['acc']*100:.1f}%)")
    print(f"  Precision  : {m['prec']:.4f}")
    print(f"  Recall     : {m['recall']:.4f}")
    print(f"  F1 Score   : {m['f1']:.4f}")
    print(f"  Pred UP    : {m['n_pred_up']:,}   Pred DOWN: {m['n_pred_down']:,}")
    print(f"{'═'*50}")


def print_feature_importance(model: EnsembleModel, top_n: int = 20) -> None:
    """In top N features theo LightGBM importance."""
    try:
        import pandas as pd
        imps = model.lgbm.feature_importances_
        cols = model.feature_cols
        df_imp = pd.DataFrame({'feature': cols, 'importance': imps})
        df_imp = df_imp.sort_values('importance', ascending=False).head(top_n)
        print(f"\n  TOP {top_n} FEATURES (LightGBM):")
        print(f"  {'Feature':<35} Importance")
        print(f"  {'─'*50}")
        for _, r in df_imp.iterrows():
            print(f"  {r['feature']:<35} {r['importance']:,.0f}")
    except Exception:
        pass


def main() -> None:
    print("╔══════════════════════════════════════════════════╗")
    print("║   BTC Promax AI v5.0  │  Model Training          ║")
    print("╚══════════════════════════════════════════════════╝")

    dl = DataLayer()
    df_5m, df_15m, df_30m, df_1h = download_data(dl, TRAIN_DAYS)

    validate_data(df_5m, '5m')

    print(f"\n  Generating features...")
    df_feat, feature_cols = generate_features(df_5m, df_15m, df_30m, df_1h, dropna=True)
    print(f"  Feature matrix: {df_feat.shape[0]:,} rows × {len(feature_cols)} features")
    print(f"  Date range: {df_feat['timestamp'].min().date()} → {df_feat['timestamp'].max().date()}")

    # Kiểm tra class balance
    y_all = (df_feat['close'].shift(-1) > df_feat['close']).astype(int)
    up_pct = y_all.mean() * 100
    print(f"  Class balance: UP={up_pct:.1f}%  DOWN={100-up_pct:.1f}%")

    # Train/test split theo thời gian (80/20)
    split_idx  = int(len(df_feat) * 0.80)
    train_df   = df_feat.iloc[:split_idx].copy()
    test_df    = df_feat.iloc[split_idx:].copy()
    print(f"\n  Train: {len(train_df):,} rows  │  Test: {len(test_df):,} rows")
    print(f"  Train end: {train_df['timestamp'].max().date()}")
    print(f"  Test start: {test_df['timestamp'].min().date()}")

    # Huấn luyện
    print(f"\n{'═'*50}")
    print("  TRAINING ENSEMBLE...")
    print(f"{'═'*50}\n")
    model = EnsembleModel()
    metrics_train = model.train(train_df, feature_cols)

    # Đánh giá test set
    print(f"\n  Evaluating on test set...")
    metrics_test = evaluate_test_set(model, test_df)
    print_metrics(metrics_test)

    # Feature importance
    print_feature_importance(model, top_n=20)

    print(f"\n  ✅ Models saved to models/")
    print(f"  Train AUC: {metrics_train.get('val_auc', 0):.4f}")
    if metrics_test:
        print(f"  Test  AUC: {metrics_test.get('auc', 0):.4f}")
        if metrics_test.get('acc', 0) >= 0.75:
            print("  🎯 Target winrate ≥75% achieved on test set!")
        else:
            print("  ⚠️  Winrate below 75% on test set — consider more data or feature tuning.")
    print()


if __name__ == '__main__':
    main()
