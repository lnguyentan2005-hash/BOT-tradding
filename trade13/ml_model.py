# === FILE: ml_model.py ===
"""
ml_model.py – Ensemble LSTM + LightGBM + XGBoost.
v5.0: LSTM có BatchNormalization, class weighting, dynamic threshold
      từ precision-recall curve, walk-forward compatible.
"""
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
import xgboost as xgb
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization, Input, Bidirectional
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from config import (
    LSTM_MODEL_PATH, LGBM_MODEL_PATH, XGB_MODEL_PATH,
    SCALER_PATH, FEATURE_COLS_PATH, THRESHOLD_PATH,
    LSTM_LOOKBACK, EPOCHS, BATCH_SIZE, LGB_PARAMS, XGB_PARAMS,
)

# Reproducibility
tf.random.set_seed(42)
np.random.seed(42)


class EnsembleModel:
    """
    Ensemble của 3 model: Bidirectional LSTM, LightGBM, XGBoost.
    Threshold tối ưu được xác định trên validation set dựa trên F1.
    """

    def __init__(self) -> None:
        self.lstm         = None
        self.lgbm         = None
        self.xgb          = None
        self.scaler       = RobustScaler()
        self.feature_cols : list[str] = []
        self.lookback     = LSTM_LOOKBACK
        self.opt_threshold: float = 0.50   # sẽ được cập nhật sau train

    # ── Architecture ─────────────────────────────────────────────────────

    def _build_lstm(self, input_shape: tuple) -> Sequential:
        model = Sequential([
            Input(shape=input_shape),
            Bidirectional(LSTM(64, return_sequences=True)),
            BatchNormalization(),
            Dropout(0.25),
            LSTM(32, return_sequences=False),
            BatchNormalization(),
            Dropout(0.20),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid'),
        ])
        model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')],
        )
        return model

    # ── Training ──────────────────────────────────────────────────────────

    def train(self, df_feat: pd.DataFrame, feature_cols: list[str]) -> dict:
        """
        Huấn luyện ensemble. Trả về dict metrics.
        df_feat phải có cột 'close'. Train/val split 85/15 theo thời gian.
        """
        self.feature_cols = feature_cols

        # Target: close[i+1] > close[i] → UP
        y_raw = (df_feat['close'].shift(-1) > df_feat['close']).astype(int).values
        X_raw = df_feat[feature_cols].values
        # Loại bỏ dòng cuối (NaN target)
        X_raw = X_raw[:-1]
        y_raw = y_raw[:-1]

        # Scale
        X_scaled = self.scaler.fit_transform(X_raw)
        joblib.dump(self.scaler, SCALER_PATH)
        joblib.dump(feature_cols, FEATURE_COLS_PATH)

        # Phân chia train/val
        split = int(len(X_scaled) * 0.85)
        X_tr_s, X_val_s = X_scaled[:split], X_scaled[split:]
        y_tr,   y_val   = y_raw[:split],    y_raw[split:]

        # Class weights cho unbalanced data
        classes = np.unique(y_tr)
        cw = compute_class_weight('balanced', classes=classes, y=y_tr)
        class_weight_dict = {c: w for c, w in zip(classes, cw)}
        print(f"  Class weights: {class_weight_dict}")

        # ── LSTM ──────────────────────────────────────────────────────────
        X_lstm_tr,  y_lstm_tr  = self._make_sequences(X_tr_s, y_tr)
        X_lstm_val, y_lstm_val = self._make_sequences(X_val_s, y_val)

        print(f"  LSTM train: {X_lstm_tr.shape[0]} seqs | val: {X_lstm_val.shape[0]} seqs")
        self.lstm = self._build_lstm((self.lookback, X_scaled.shape[1]))
        self.lstm.fit(
            X_lstm_tr, y_lstm_tr,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_lstm_val, y_lstm_val),
            class_weight=class_weight_dict,
            callbacks=[
                EarlyStopping(patience=8, restore_best_weights=True, monitor='val_auc', mode='max'),
                ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-5, monitor='val_auc', mode='max'),
            ],
            verbose=1,
        )
        self.lstm.save(LSTM_MODEL_PATH)

        # ── LightGBM & XGBoost ────────────────────────────────────────────
        # Dùng từ lookback trở đi để align với LSTM
        X_lgb_tr  = X_tr_s[self.lookback:]
        y_lgb_tr  = y_tr[self.lookback:]
        X_lgb_val = X_val_s[self.lookback:]
        y_lgb_val = y_val[self.lookback:]

        print("  Training LightGBM...")
        self.lgbm = lgb.LGBMClassifier(**LGB_PARAMS)
        self.lgbm.fit(
            X_lgb_tr, y_lgb_tr,
            eval_set=[(X_lgb_val, y_lgb_val)],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
        )
        joblib.dump(self.lgbm, LGBM_MODEL_PATH)

        print("  Training XGBoost...")
        pos_ratio = (y_lgb_tr == 0).sum() / max((y_lgb_tr == 1).sum(), 1)
        xgb_params = {**XGB_PARAMS, 'scale_pos_weight': pos_ratio}
        self.xgb = xgb.XGBClassifier(**xgb_params)
        self.xgb.fit(
            X_lgb_tr, y_lgb_tr,
            eval_set=[(X_lgb_val, y_lgb_val)],
            verbose=False,
        )
        joblib.dump(self.xgb, XGB_MODEL_PATH)

        # ── Tối ưu threshold trên validation set ─────────────────────────
        self.opt_threshold = self._optimize_threshold(
            X_lstm_val, y_lstm_val, X_lgb_val, y_lgb_val
        )
        joblib.dump(self.opt_threshold, THRESHOLD_PATH)
        print(f"  Optimal threshold: {self.opt_threshold:.3f}")

        # ── Metrics ───────────────────────────────────────────────────────
        val_probs = self._ensemble_proba(X_lstm_val, X_lgb_val)
        auc = roc_auc_score(y_val[self.lookback:], val_probs)
        preds = (val_probs >= self.opt_threshold).astype(int)
        acc = np.mean(preds == y_val[self.lookback:])
        print(f"  Validation AUC: {auc:.4f}  Accuracy@threshold: {acc:.4f}")
        return {'val_auc': auc, 'val_acc': acc, 'threshold': self.opt_threshold}

    def _make_sequences(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        Xs, ys = [], []
        for i in range(self.lookback, len(X)):
            Xs.append(X[i - self.lookback:i])
            ys.append(y[i])
        return np.array(Xs), np.array(ys)

    def _ensemble_proba(self, X_lstm: np.ndarray, X_flat: np.ndarray) -> np.ndarray:
        """Tính trung bình 3 model. X_lstm: (N, lookback, feats), X_flat: (N, feats)."""
        lstm_probs = self.lstm.predict(X_lstm, verbose=0).flatten()
        lgbm_probs = self.lgbm.predict_proba(X_flat)[:, 1]
        xgb_probs  = self.xgb.predict_proba(X_flat)[:, 1]
        return (lstm_probs + lgbm_probs + xgb_probs) / 3

    def _optimize_threshold(
        self,
        X_lstm_val: np.ndarray, y_lstm_val: np.ndarray,
        X_flat_val: np.ndarray, y_flat_val: np.ndarray,
    ) -> float:
        """Chọn threshold tối đa hoá F1 trên validation."""
        min_len = min(len(y_lstm_val), len(y_flat_val))
        if min_len == 0:
            return 0.50
        probs = self._ensemble_proba(X_lstm_val[:min_len], X_flat_val[:min_len])
        y_val = y_lstm_val[:min_len]
        precision, recall, thresholds = precision_recall_curve(y_val, probs)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        best_idx = np.argmax(f1)
        thresh = float(thresholds[min(best_idx, len(thresholds) - 1)])
        return max(0.50, min(0.75, thresh))   # clamp trong [0.50, 0.75]

    # ── Load ─────────────────────────────────────────────────────────────

    def load_models(self) -> None:
        self.scaler       = joblib.load(SCALER_PATH)
        self.feature_cols = joblib.load(FEATURE_COLS_PATH)
        self.lstm         = load_model(LSTM_MODEL_PATH)
        self.lgbm         = joblib.load(LGBM_MODEL_PATH)
        self.xgb          = joblib.load(XGB_MODEL_PATH)
        try:
            self.opt_threshold = joblib.load(THRESHOLD_PATH)
        except FileNotFoundError:
            self.opt_threshold = 0.55

    # ── Inference ────────────────────────────────────────────────────────

    def predict_proba(
        self, df_feat: pd.DataFrame
    ) -> tuple[float | None, dict[str, float] | None]:
        """
        Trả về (avg_prob_up, {'lstm': .., 'lgbm': .., 'xgb': ..}) 
        hoặc (None, None) nếu không đủ data.
        """
        if not self.feature_cols:
            return None, None
        try:
            data   = df_feat[self.feature_cols].values
            scaled = self.scaler.transform(data)
        except Exception:
            return None, None

        if len(scaled) < self.lookback:
            return None, None

        X_seq  = scaled[-self.lookback:].reshape(1, self.lookback, -1)
        X_flat = scaled[-1].reshape(1, -1)

        try:
            lstm_prob = float(self.lstm.predict(X_seq, verbose=0)[0][0])
            lgbm_prob = float(self.lgbm.predict_proba(X_flat)[0][1])
            xgb_prob  = float(self.xgb.predict_proba(X_flat)[0][1])
        except Exception as exc:
            print(f"[EnsembleModel] inference error: {exc}")
            return None, None

        avg = (lstm_prob + lgbm_prob + xgb_prob) / 3
        return avg, {'lstm': lstm_prob, 'lgbm': lgbm_prob, 'xgb': xgb_prob}
