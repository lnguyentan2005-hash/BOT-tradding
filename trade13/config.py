# === FILE: config.py ===
"""
config.py – Cấu hình tập trung cho BTC Promax AI v5.0
Tất cả hằng số, đường dẫn, hyperparameter đều nằm ở đây.
"""
import os

# ==================== Exchange ====================
SYMBOL          = 'BTC/USDT'
EXCHANGE_ID     = 'binance'

# ==================== Timeframes ====================
TIMEFRAME_PRIMARY = '5m'
TIMEFRAME_15M     = '15m'
TIMEFRAME_30M     = '30m'
TIMEFRAME_1H      = '1h'

LOOKBACK_LIVE = 350      # số nến 5m cho live (cần đủ cho LSTM + indicators)

# ==================== Paths ====================
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR  = os.path.join(BASE_DIR, 'data')
LOG_DIR   = os.path.join(BASE_DIR, 'logs')

for d in [MODEL_DIR, DATA_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

LSTM_MODEL_PATH    = os.path.join(MODEL_DIR, 'lstm_model.keras')
LGBM_MODEL_PATH    = os.path.join(MODEL_DIR, 'lgbm_model.pkl')
XGB_MODEL_PATH     = os.path.join(MODEL_DIR, 'xgb_model.pkl')
SCALER_PATH        = os.path.join(MODEL_DIR, 'scaler.pkl')
FEATURE_COLS_PATH  = os.path.join(MODEL_DIR, 'feature_cols.pkl')
THRESHOLD_PATH     = os.path.join(MODEL_DIR, 'opt_threshold.pkl')

TRAIN_DATA_FILE_5M  = os.path.join(DATA_DIR, 'btc_5m.csv')
TRAIN_DATA_FILE_15M = os.path.join(DATA_DIR, 'btc_15m.csv')
TRAIN_DATA_FILE_30M = os.path.join(DATA_DIR, 'btc_30m.csv')
TRAIN_DATA_FILE_1H  = os.path.join(DATA_DIR, 'btc_1h.csv')

TRADE_LOG_CSV     = os.path.join(LOG_DIR, 'trade_log.csv')
BACKTEST_LOG_CSV  = os.path.join(LOG_DIR, 'backtest_log.csv')

# ==================== Training ====================
TRAIN_DAYS            = 180    # ngày lịch sử cho train
MIN_CANDLES_FOR_TRAIN = 1500
LSTM_LOOKBACK         = 60
BATCH_SIZE            = 64
EPOCHS                = 60

LGB_PARAMS = {
    'n_estimators'   : 500,
    'num_leaves'     : 63,
    'learning_rate'  : 0.03,
    'min_child_samples': 30,
    'subsample'      : 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha'      : 0.1,
    'reg_lambda'     : 0.1,
    'class_weight'   : 'balanced',
    'random_state'   : 42,
    'verbose'        : -1,
}
XGB_PARAMS = {
    'n_estimators'  : 500,
    'max_depth'     : 5,
    'learning_rate' : 0.03,
    'subsample'     : 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha'     : 0.1,
    'reg_lambda'    : 0.1,
    'scale_pos_weight': 1.0,   # sẽ được tính động trong train.py
    'eval_metric'   : 'logloss',
    'random_state'  : 42,
    'verbosity'     : 0,
}

# ==================== Decision Thresholds ====================
ADX_THRESHOLD         = 22      # lọc sideway
SLOPE_THRESHOLD       = 0.025   # % EMA20 slope tối thiểu
MIN_TREND_FRAMES      = 3       # tối thiểu 3/4 khung đồng thuận
CONSENSUS_RATIO       = 0.67    # 2/3 model đồng thuận
ML_PROB_THRESHOLD     = 0.60    # ngưỡng xác suất ML
ML_HIGH_CONF          = 0.68    # ngưỡng HIGH confidence
TECH_SCORE_THRESHOLD  = 2.5     # ngưỡng TA score bình thường
TECH_SCORE_CAUTIOUS   = 3.5     # ngưỡng TA score cautious mode
VOLATILITY_MAX_ATR    = 0.35    # bỏ qua nến cực kỳ biến động (ATR% > x)

# ==================== Risk ====================
MAX_CONSECUTIVE_LOSSES = 2
CAUTIOUS_CANDLES       = 4      # nến phải chờ sau khi hit max losses
DAILY_LOSS_LIMIT       = 5      # dừng ngày nếu thua quá nhiều lệnh

# ==================== Session Windows (UTC) ====================
# Phiên có độ biến động tốt cho 5m scalping
ACTIVE_SESSIONS = [
    (1,  9),   # Asian + overlap
    (7, 16),   # London
    (13, 21),  # New York
]

# ==================== Backtest ====================
# Sửa 2 dòng dưới để thay đổi khoảng thời gian backtest
BACKTEST_START = '2026-04-01'
BACKTEST_END   = '2026-04-29'
TRAIN_START    = '2025-09-01'
TRAIN_END      = '2025-12-31'

# ==================== Logging ====================
TIMEZONE = 'Asia/Ho_Chi_Minh'

# ==================== Telegram (optional) ====================
TELEGRAM_BOT_TOKEN = ''
TELEGRAM_CHAT_ID   = ''
