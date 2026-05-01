# ========== CẤU HÌNH CHO TRAIN & BACKTEST ==========
SYMBOL = "BTCUSDT"
INTERVAL = "5m"
EXTRA_TIMEFRAMES = ['15m', '1h', '4h', '1d']
LOOKBACK_WINDOW = 15
STRONG_PROB_THRESHOLD = 0.69
USE_TREND_FILTER = False
EMA_TREND_PERIOD = 20

# File ngày tháng cho train
DATES_FILE = "datetrain.txt"

# Thư mục lưu trữ riêng cho train
CACHE_DIR = "data_train/cache"
MODEL_FILE = "data_train/models/btc_advanced_model.pkl"

# Tên file kết quả backtest
OUTPUT_EXCEL = "prediction_results_advanced.xlsx"

# Tham số LightGBM
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