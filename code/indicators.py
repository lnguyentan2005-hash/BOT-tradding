import pandas as pd
import numpy as np

# ===================== CÁC CHỈ BÁO =====================
def compute_adx(df, period=14):
    high, low, close = df['high'], df['low'], df['close']
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9))
    adx = dx.rolling(period).mean()
    return adx, plus_di, minus_di

def compute_supertrend(df, period=7, multiplier=3):
    high, low, close = df['high'], df['low'], df['close']
    hl2 = (high + low) / 2
    atr = (high - low).rolling(period).mean()
    upperband = hl2 + multiplier * atr
    lowerband = hl2 - multiplier * atr
    supertrend = pd.Series(1, index=df.index)
    direction = 1
    for i in range(1, len(df)):
        if close.iloc[i] > upperband.iloc[i-1]:
            direction = 1
        elif close.iloc[i] < lowerband.iloc[i-1]:
            direction = -1
        if direction == 1 and lowerband.iloc[i] < lowerband.iloc[i-1]:
            lowerband.iloc[i] = lowerband.iloc[i-1]
        if direction == -1 and upperband.iloc[i] > upperband.iloc[i-1]:
            upperband.iloc[i] = upperband.iloc[i-1]
        supertrend.iloc[i] = direction
    supertrend_line = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        supertrend_line.iloc[i] = lowerband.iloc[i] if supertrend.iloc[i] == 1 else upperband.iloc[i]
    return supertrend, supertrend_line

def compute_parabolic_sar(df, af=0.02, max_af=0.2):
    high, low = df['high'], df['low']
    sar = pd.Series(index=df.index, dtype=float)
    ep = pd.Series(index=df.index, dtype=float)
    af_val = af
    direction = 1
    sar.iloc[0] = low.iloc[0]
    ep.iloc[0] = high.iloc[0]
    for i in range(1, len(df)):
        prev_sar = sar.iloc[i-1]
        prev_ep = ep.iloc[i-1]
        if direction == 1:
            sar.iloc[i] = prev_sar + af_val * (prev_ep - prev_sar)
            if high.iloc[i] > prev_ep:
                ep.iloc[i] = high.iloc[i]
                af_val = min(af_val + af, max_af)
            else:
                ep.iloc[i] = prev_ep
            if low.iloc[i] < sar.iloc[i]:
                direction = -1
                af_val = af
                sar.iloc[i] = max(high.iloc[i-1], high.iloc[i])
                ep.iloc[i] = low.iloc[i]
        else:
            sar.iloc[i] = prev_sar - af_val * (prev_sar - prev_ep)
            if low.iloc[i] < prev_ep:
                ep.iloc[i] = low.iloc[i]
                af_val = min(af_val + af, max_af)
            else:
                ep.iloc[i] = prev_ep
            if high.iloc[i] > sar.iloc[i]:
                direction = 1
                af_val = af
                sar.iloc[i] = min(low.iloc[i-1], low.iloc[i])
                ep.iloc[i] = high.iloc[i]
    return sar

def compute_ichimoku(df, tenkan=9, kijun=26, senkou_b=52):
    high, low = df['high'], df['low']
    tenkan_sen = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2
    kijun_sen = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2
    senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
    senkou_b = ((high.rolling(senkou_b).max() + low.rolling(senkou_b).min()) / 2).shift(kijun)
    return tenkan_sen, kijun_sen, senkou_a, senkou_b

def compute_mfi(df, period=14):
    tp = (df['high'] + df['low'] + df['close']) / 3
    money_flow = tp * df['volume']
    positive_flow = money_flow.where(tp > tp.shift(1), 0)
    negative_flow = money_flow.where(tp < tp.shift(1), 0)
    positive_mf = positive_flow.rolling(period).sum()
    negative_mf = negative_flow.rolling(period).sum()
    mfi = 100 - (100 / (1 + positive_mf / negative_mf))
    return mfi

def compute_cmf(df, period=20):
    money_flow_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-9)
    money_flow_volume = money_flow_multiplier * df['volume']
    cmf = money_flow_volume.rolling(period).sum() / df['volume'].rolling(period).sum()
    return cmf

def compute_vwap(df):
    """VWAP tích lũy trong ngày (reset mỗi ngày)"""
    tp = (df['high'] + df['low'] + df['close']) / 3
    df_vwap = tp * df['volume']
    # groupby ngày (theo index)
    grouped = df.groupby(df.index.date)
    cum_vp = grouped['close'].transform(lambda x: (tp[x.index] * df['volume'][x.index]).cumsum())
    cum_vol = grouped['volume'].transform('cumsum')
    vwap = cum_vp / cum_vol
    return vwap

def compute_ad_line(df):
    """A/D Line tích lũy trong ngày"""
    money_flow_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-9)
    money_flow_volume = money_flow_multiplier * df['volume']
    # Tính A/D line trong ngày
    grouped = df.groupby(df.index.date)
    ad_line = grouped['close'].transform(lambda x: money_flow_volume[x.index].cumsum())
    return ad_line

def compute_eom(df, period=14):
    distance_moved = ((df['high'] + df['low']) / 2) - ((df['high'].shift(1) + df['low'].shift(1)) / 2)
    box_ratio = (df['volume'] / 1e6) / (df['high'] - df['low'] + 1e-9)
    eom = distance_moved / box_ratio
    eom_sma = eom.rolling(period).mean()
    return eom_sma

def compute_keltner_channels(df, period=20, multiplier=2):
    tp = (df['high'] + df['low'] + df['close']) / 3
    ema_tp = tp.ewm(span=period, adjust=False).mean()
    atr = (df['high'] - df['low']).rolling(period).mean()
    upper = ema_tp + multiplier * atr
    lower = ema_tp - multiplier * atr
    return upper, lower

def compute_donchian_channels(df, period=20):
    upper = df['high'].rolling(period).max()
    lower = df['low'].rolling(period).min()
    middle = (upper + lower) / 2
    return upper, lower, middle

def compute_historical_volatility(df, period=20):
    returns = df['close'].pct_change()
    hv = returns.rolling(period).std() * np.sqrt(365 * 24 * 12)
    return hv

def compute_ulcer_index(df, period=14):
    close = df['close']
    max_close = close.rolling(period).max()
    drawdown = 100 * (close - max_close) / max_close
    squared_drawdown = drawdown ** 2
    ulcer = np.sqrt(squared_drawdown.rolling(period).mean())
    return ulcer

def detect_candlestick_patterns(df):
    close, open_ = df['close'], df['open']
    body = abs(close - open_)
    doji = body / (df['high'] - df['low'] + 1e-9) < 0.1
    morning_star = (
        (close.shift(2) < open_.shift(2)) &
        (body.shift(1) < body.shift(2) * 0.5) &
        (close > open_) &
        (close > (open_.shift(2) + close.shift(2)) / 2)
    )
    evening_star = (
        (close.shift(2) > open_.shift(2)) &
        (body.shift(1) < body.shift(2) * 0.5) &
        (close < open_) &
        (close < (open_.shift(2) + close.shift(2)) / 2)
    )
    white_soldiers = (
        (close > open_) &
        (close.shift(1) > open_.shift(1)) &
        (close.shift(2) > open_.shift(2)) &
        (close > close.shift(1)) &
        (close.shift(1) > close.shift(2))
    )
    black_crows = (
        (close < open_) &
        (close.shift(1) < open_.shift(1)) &
        (close.shift(2) < open_.shift(2)) &
        (close < close.shift(1)) &
        (close.shift(1) < close.shift(2))
    )
    return morning_star.astype(int), evening_star.astype(int), white_soldiers.astype(int), black_crows.astype(int)

# ===================== HÀM TÍNH TẤT CẢ CHỈ BÁO =====================
def compute_indicators(df):
    data = df.copy()
    # ---- EMA ----
    data['ema5'] = data['close'].ewm(span=5, adjust=False).mean()
    data['ema10'] = data['close'].ewm(span=10, adjust=False).mean()
    data['ema20'] = data['close'].ewm(span=20, adjust=False).mean()
    data['ema50'] = data['close'].ewm(span=50, adjust=False).mean()
    # ---- RSI ----
    delta = data['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    data['rsi'] = 100 - (100 / (1 + rs))
    # ---- MACD ----
    exp12 = data['close'].ewm(span=12, adjust=False).mean()
    exp26 = data['close'].ewm(span=26, adjust=False).mean()
    data['macd'] = exp12 - exp26
    data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
    data['macd_hist'] = data['macd'] - data['macd_signal']
    # ---- Stochastic ----
    low_14 = data['low'].rolling(14).min()
    high_14 = data['high'].rolling(14).max()
    data['stoch_k'] = 100 * (data['close'] - low_14) / (high_14 - low_14 + 1e-9)
    data['stoch_d'] = data['stoch_k'].rolling(3).mean()
    # ---- ATR ----
    high_low = data['high'] - data['low']
    high_close = np.abs(data['high'] - data['close'].shift())
    low_close = np.abs(data['low'] - data['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data['atr'] = tr.rolling(14).mean()
    # ---- Bollinger Bands ----
    sma20 = data['close'].rolling(20).mean()
    std20 = data['close'].rolling(20).std()
    data['bb_upper'] = sma20 + 2 * std20
    data['bb_lower'] = sma20 - 2 * std20
    data['bb_width'] = data['bb_upper'] - data['bb_lower']
    data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_width'] + 1e-9)
    # ---- CCI ----
    tp = (data['high'] + data['low'] + data['close']) / 3
    sma_tp = tp.rolling(20).mean()
    mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    data['cci'] = (tp - sma_tp) / (0.015 * mad + 1e-9)
    # ---- OBV (reset theo ngày) ----
    data['obv_daily'] = data.groupby(data.index.date)['close'].transform(
        lambda x: (np.sign(x.diff()) * data.loc[x.index, 'volume']).fillna(0).cumsum()
    )
    # ---- Volume Ratio ----
    data['vol_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
    # ---- Biến động cơ bản ----
    data['returns'] = data['close'].pct_change()
    data['hl_ratio'] = (data['high'] - data['low']) / (data['close'] + 1e-9)
    data['body'] = data['close'] - data['open']
    data['momentum'] = data['close'] - data['close'].shift(5)
    # ---- Mẫu nến ----
    data['doji'] = (abs(data['body']) / (data['high'] - data['low'] + 1e-9) < 0.1).astype(int)
    data['hammer'] = ((data['high'] - data['low']) > 3 * abs(data['body'])) & \
                     (data['close'] > data['open']) & ((data['open'] - data['low']) > 2 * abs(data['body']))
    data['hammer'] = data['hammer'].astype(int)
    data['shooting_star'] = ((data['high'] - data['low']) > 3 * abs(data['body'])) & \
                            (data['close'] < data['open']) & ((data['high'] - data['close']) > 2 * abs(data['body']))
    data['shooting_star'] = data['shooting_star'].astype(int)
    data['engulfing_up'] = ((data['close'] > data['open']) & (data['close'].shift() < data['open'].shift()) &
                            (data['close'] > data['close'].shift()) & (data['open'] < data['open'].shift())).astype(int)
    data['engulfing_down'] = ((data['close'] < data['open']) & (data['close'].shift() > data['open'].shift()) &
                              (data['close'] < data['close'].shift()) & (data['open'] > data['open'].shift())).astype(int)
    # ---- Các chỉ báo mới ----
    adx, plus_di, minus_di = compute_adx(df)
    data['adx'] = adx
    data['plus_di'] = plus_di
    data['minus_di'] = minus_di

    supertrend, supertrend_line = compute_supertrend(df)
    data['supertrend_dir'] = supertrend
    data['supertrend_line'] = supertrend_line

    data['psar'] = compute_parabolic_sar(df)

    tenkan, kijun, senkou_a, senkou_b = compute_ichimoku(df)
    data['tenkan'] = tenkan
    data['kijun'] = kijun
    data['senkou_a'] = senkou_a
    data['senkou_b'] = senkou_b

    data['mfi'] = compute_mfi(df)
    data['cmf'] = compute_cmf(df)
    data['vwap'] = compute_vwap(df)
    data['ad_line'] = compute_ad_line(df)
    data['eom'] = compute_eom(df)

    kc_upper, kc_lower = compute_keltner_channels(df)
    data['kc_upper'] = kc_upper
    data['kc_lower'] = kc_lower
    data['kc_position'] = (data['close'] - kc_lower) / (kc_upper - kc_lower + 1e-9)

    dc_upper, dc_lower, dc_middle = compute_donchian_channels(df)
    data['dc_upper'] = dc_upper
    data['dc_lower'] = dc_lower
    data['dc_middle'] = dc_middle
    data['dc_position'] = (data['close'] - dc_lower) / (dc_upper - dc_lower + 1e-9)

    data['hist_vol'] = compute_historical_volatility(df)
    data['ulcer_index'] = compute_ulcer_index(df)

    morning, evening, three_white, three_black = detect_candlestick_patterns(df)
    data['morning_star'] = morning
    data['evening_star'] = evening
    data['three_white_soldiers'] = three_white
    data['three_black_crows'] = three_black

    data['hour'] = data.index.hour
    data['minute'] = data.index.minute
    data['day_of_week'] = data.index.dayofweek
    data['day_of_month'] = data.index.day
    data['month'] = data.index.month

    data.dropna(inplace=True)
    return data

# ===================== HỢP NHẤT KHUNG LỚN (đã tối ưu) =====================
def merge_with_higher_tf(df_5m, dict_higher_tf):
    df_merged = df_5m.copy()
    df_merged.sort_index(inplace=True)
    # Tạo cột merge_time chỉ một lần
    df_merged['merge_time'] = df_merged.index
    merge_frames = [df_merged]
    for tf, df_htf in dict_higher_tf.items():
        indicator_cols = [col for col in df_htf.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        if not indicator_cols:
            continue
        df_htf_sorted = df_htf[indicator_cols].sort_index()
        merged = pd.merge_asof(
            df_merged[['merge_time']].sort_values('merge_time'),
            df_htf_sorted,
            left_on='merge_time',
            right_index=True,
            direction='backward',
            allow_exact_matches=False
        )
        rename_dict = {col: f"{col}_{tf}" for col in indicator_cols}
        merged.rename(columns=rename_dict, inplace=True)
        merged.drop(columns=['merge_time'], inplace=True, errors='ignore')
        merge_frames.append(merged)
    final = pd.concat(merge_frames, axis=1)
    final.drop(columns=['merge_time'], inplace=True, errors='ignore')
    final.dropna(inplace=True)
    return final