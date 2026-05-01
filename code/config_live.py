# ========== CẤU HÌNH CHO LIVEBOT ==========
SYMBOL = "BTCUSDT"
INTERVAL = "5m"
EXTRA_TIMEFRAMES = ['15m', '1h', '4h', '1d']
LOOKBACK_WINDOW = 15
STRONG_PROB_THRESHOLD = 0.63

# Bộ lọc xu hướng EMA
USE_TREND_FILTER = False          # Bật lọc EMA20
EMA_TREND_PERIOD = 20            # Chu kỳ EMA để lọc

# Thư mục lưu trữ riêng cho livebot
CACHE_DIR = "data_live/cache"
MODEL_FILE = "data_live/models/btc_advanced_model.pkl"

# File log
EXCEL_LOG = "live_trade_log.xlsx"

# Tham số LightGBM (chỉ dùng khi load model, không cần train)
LGBM_PARAMS = dict(
    n_estimators=200,
    learning_rate=0.02,
    num_leaves=31,
    max_depth=6,
    min_child_samples=200,
    subsample=0.7,
    colsample_bytree=0.4,
    reg_alpha=0.3,
    reg_lambda=0.3,
    class_weight='balanced',
    random_state=42,
    verbose=-1
)