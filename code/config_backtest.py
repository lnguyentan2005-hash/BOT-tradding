# ========== CẤU HÌNH DÀNH RIÊNG CHO BACKTEST ==========
SYMBOL = "BTCUSDT"
INTERVAL = "5m"
EXTRA_TIMEFRAMES = ['15m', '1h', '4h', '1d']
LOOKBACK_WINDOW = 15
STRONG_PROB_THRESHOLD = 0.63          # Hạ ngưỡng để tăng số lệnh
USE_TREND_FILTER = False             # Tắt lọc EMA20 để không bỏ lỡ tín hiệu
EMA_TREND_PERIOD = 20                # Không dùng nếu USE_TREND_FILTER = False

# File ngày test
TEST_DATE_FILE = "testdate.txt"

# Thư mục cache & model (dùng chung với train)
CACHE_DIR = "data_train/cache"
MODEL_FILE = "data_train/models/btc_advanced_model.pkl"

# File kết quả backtest
OUTPUT_BACKTEST = "backtest_results.xlsx"

# Tham số LightGBM (chỉ để load model, không dùng train)
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