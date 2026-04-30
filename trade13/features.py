# === FILE: features.py ===
"""
features.py – Feature engineering đa khung thời gian cho BTC/USDT 5m.
Bổ sung so với v4: VWAP distance, candle patterns, market structure (HH/LL),
session encoding, momentum divergence, lagged features.
"""
import pandas as pd
import numpy as np
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ensure_utc_ns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).astype('datetime64[ns, UTC]')
    return df


def _ema20_slope(df: pd.DataFrame, col_name: str) -> pd.Series:
    ema = EMAIndicator(df['close'], 20).ema_indicator()
    return ema.pct_change() * 100


# ── Candle pattern features ───────────────────────────────────────────────────

def _add_candle_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Thêm các pattern nến cơ bản dưới dạng +1 / -1 / 0."""
    o, h, l, c = df['open'], df['high'], df['low'], df['close']
    body   = (c - o).abs()
    rng    = h - l + 1e-10
    upper  = h - np.maximum(c, o)
    lower  = np.minimum(c, o) - l
    body_pct = body / rng

    # Hammer / Shooting Star
    df['pat_hammer']       = ((lower > body * 2) & (upper < body * 0.3) & (body_pct < 0.4)).astype(np.int8)
    df['pat_shooting_star']= ((upper > body * 2) & (lower < body * 0.3) & (body_pct < 0.4)).astype(np.int8)

    # Doji
    df['pat_doji']         = (body_pct < 0.1).astype(np.int8)

    # Engulfing (bullish / bearish)
    prev_body = body.shift(1)
    prev_bull = (df['close'].shift(1) > df['open'].shift(1))
    bull_eng  = (~prev_bull) & (c > o) & (c > df['open'].shift(1)) & (o < df['close'].shift(1)) & (body > prev_body)
    bear_eng  = (prev_bull) & (c < o) & (c < df['open'].shift(1)) & (o > df['close'].shift(1)) & (body > prev_body)
    df['pat_bull_engulf'] = bull_eng.astype(np.int8)
    df['pat_bear_engulf'] = bear_eng.astype(np.int8)

    return df


# ── Market structure ──────────────────────────────────────────────────────────

def _add_market_structure(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Đơn giản hóa HH/HL/LH/LL: 
    - structure_bull (+1) = higher high AND higher low trong `window` nến trước
    - structure_bear (-1) = lower high AND lower low
    - 0 = không xác định
    """
    rh = df['high'].rolling(window).max()
    rl = df['low'].rolling(window).min()
    ph = df['high'].rolling(window).max().shift(window)
    pl = df['low'].rolling(window).min().shift(window)

    bull = ((rh > ph) & (rl > pl)).fillna(False)
    bear = ((rh < ph) & (rl < pl)).fillna(False)
    df['structure'] = np.where(bull, 1, np.where(bear, -1, 0)).astype(np.int8)
    return df


# ── Support / Resistance proximity ───────────────────────────────────────────

def _add_sr_proximity(df: pd.DataFrame, lookback: int = 50) -> pd.DataFrame:
    """
    Tính khoảng cách % từ close đến mức high / low của `lookback` nến trước.
    Gần support → xét UP; gần resistance → xét DOWN.
    """
    df['sr_high'] = df['high'].rolling(lookback).max().shift(1)
    df['sr_low']  = df['low'].rolling(lookback).min().shift(1)
    df['dist_to_resistance'] = (df['sr_high'] - df['close']) / df['close'] * 100
    df['dist_to_support']    = (df['close'] - df['sr_low'])  / df['close'] * 100
    return df


# ── Momentum divergence ───────────────────────────────────────────────────────

def _add_divergence(df: pd.DataFrame) -> pd.DataFrame:
    """
    RSI divergence đơn giản: giá tăng nhưng RSI giảm (bearish) và ngược lại.
    Giá trị: +1 bullish div, -1 bearish div, 0 không.
    """
    price_up  = df['close'] > df['close'].shift(3)
    rsi_up    = df['rsi'] > df['rsi'].shift(3)
    bull_div  = (~price_up) & rsi_up
    bear_div  = price_up & (~rsi_up)
    df['rsi_divergence'] = np.where(bull_div, 1, np.where(bear_div, -1, 0)).astype(np.int8)
    return df


# ── Main feature generator ────────────────────────────────────────────────────

def generate_features(
    df_5m:  pd.DataFrame,
    df_15m: pd.DataFrame | None = None,
    df_30m: pd.DataFrame | None = None,
    df_1h:  pd.DataFrame | None = None,
    dropna: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Trả về (DataFrame with all features, list[feature_col_names]).
    Cột meta (timestamp, ohlcv) không nằm trong feature_cols.
    """
    df = _ensure_utc_ns(df_5m.copy())
    close = df['close']

    # ── EMA & slope ──────────────────────────────────────────────────────
    df['ema5']           = EMAIndicator(close, 5).ema_indicator()
    df['ema10']          = EMAIndicator(close, 10).ema_indicator()
    df['ema20']          = EMAIndicator(close, 20).ema_indicator()
    df['ema50']          = EMAIndicator(close, 50).ema_indicator()
    df['ema20_slope_pct']= df['ema20'].pct_change() * 100
    df['ema5_above_20']  = (df['ema5'] > df['ema20']).astype(np.int8)
    df['ema10_above_20'] = (df['ema10'] > df['ema20']).astype(np.int8)
    df['price_above_50'] = (close > df['ema50']).astype(np.int8)

    # ── RSI ──────────────────────────────────────────────────────────────
    df['rsi']         = RSIIndicator(close, 14).rsi()
    df['rsi_change']  = df['rsi'].diff(3)          # momentum của RSI

    # ── MACD ─────────────────────────────────────────────────────────────
    macd              = MACD(close, 26, 12, 9)
    df['macd']        = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff']   = macd.macd_diff()
    df['macd_cross']  = np.sign(df['macd_diff']).diff().fillna(0).astype(np.int8)  # +2/-2 tại điểm giao cắt

    # ── Stochastic ────────────────────────────────────────────────────────
    stoch             = StochasticOscillator(df['high'], df['low'], close, 14, 3)
    df['stoch_k']     = stoch.stoch()
    df['stoch_d']     = stoch.stoch_signal()
    df['stoch_diff']  = df['stoch_k'] - df['stoch_d']

    # ── Bollinger Bands ───────────────────────────────────────────────────
    bb                = BollingerBands(close, 20, 2)
    df['bb_upper']    = bb.bollinger_hband()
    df['bb_lower']    = bb.bollinger_lband()
    df['bb_width']    = df['bb_upper'] - df['bb_lower']
    df['bb_pct']      = (close - df['bb_lower']) / (df['bb_width'] + 1e-10)
    df['bb_squeeze']  = (df['bb_width'] / close * 100 < 1.5).astype(np.int8)  # squeeze

    # ── ATR & volatility ─────────────────────────────────────────────────
    atr               = AverageTrueRange(df['high'], df['low'], close, 14)
    df['atr']         = atr.average_true_range()
    df['atr_pct']     = df['atr'] / close * 100
    df['atr_ratio']   = df['atr_pct'] / df['atr_pct'].rolling(20).mean()  # regime vol

    # ── ADX ───────────────────────────────────────────────────────────────
    adx_ind           = ADXIndicator(df['high'], df['low'], close, 14)
    df['adx']         = adx_ind.adx()
    df['adx_pos']     = adx_ind.adx_pos()  # +DI
    df['adx_neg']     = adx_ind.adx_neg()  # -DI
    df['di_diff']     = df['adx_pos'] - df['adx_neg']  # sign = hướng xu hướng

    # ── Volume ────────────────────────────────────────────────────────────
    df['volume_ma20']  = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_ma20'] + 1e-10)
    df['vol_spike']    = (df['volume_ratio'] > 2.0).astype(np.int8)

    # ── VWAP (đã tính sẵn trong DataLayer nếu có, tính lại nếu thiếu) ────
    if 'vwap' not in df.columns:
        typical        = (df['high'] + df['low'] + close) / 3
        cum_tp         = (typical * df['volume']).rolling(20, min_periods=1).sum()
        cum_vol        = df['volume'].rolling(20, min_periods=1).sum()
        df['vwap']     = cum_tp / (cum_vol + 1e-10)
    df['vwap_dist']    = (close - df['vwap']) / (df['vwap'] + 1e-10) * 100

    # ── Returns & momentum ────────────────────────────────────────────────
    df['returns_1']    = close.pct_change()
    df['returns_3']    = close.pct_change(3)
    df['returns_5']    = close.pct_change(5)
    df['returns_10']   = close.pct_change(10)
    df['volatility_10']= df['returns_1'].rolling(10).std()
    df['momentum']     = close - close.shift(5)
    df['high_low_ratio']  = (df['high'] - df['low']) / close
    df['upper_shadow']    = df['high'] - np.maximum(close, df['open'])
    df['lower_shadow']    = np.minimum(close, df['open']) - df['low']
    df['candle_body_pct'] = (close - df['open']).abs() / (df['high'] - df['low'] + 1e-10)

    # ── Lagged key features (giảm look-ahead bias) ───────────────────────
    for lag in [1, 2, 3]:
        df[f'rsi_lag{lag}']   = df['rsi'].shift(lag)
        df[f'macd_lag{lag}']  = df['macd_diff'].shift(lag)

    # ── Time encoding ─────────────────────────────────────────────────────
    hours               = df['timestamp'].dt.hour
    minutes             = df['timestamp'].dt.minute
    df['hour_sin']      = np.sin(2 * np.pi * hours / 24)
    df['hour_cos']      = np.cos(2 * np.pi * hours / 24)
    df['min_sin']       = np.sin(2 * np.pi * minutes / 60)
    df['is_session_asian']  = ((hours >= 1) & (hours < 9)).astype(np.int8)
    df['is_session_london'] = ((hours >= 7) & (hours < 16)).astype(np.int8)
    df['is_session_ny']     = ((hours >= 13) & (hours < 21)).astype(np.int8)

    # ── Candle patterns ───────────────────────────────────────────────────
    df = _add_candle_patterns(df)

    # ── Market structure ──────────────────────────────────────────────────
    df = _add_market_structure(df, window=10)

    # ── Support / Resistance ──────────────────────────────────────────────
    df = _add_sr_proximity(df, lookback=50)

    # ── Merge slopes từ higher timeframes ────────────────────────────────
    df = _merge_tf_slope(df, df_15m, 'ema20_15m_slope_pct')
    df = _merge_tf_slope(df, df_30m, 'ema20_30m_slope_pct')
    df = _merge_tf_slope(df, df_1h,  'ema20_1h_slope_pct')

    # ── RSI divergence (sau khi có rsi và multi-tf) ───────────────────────
    df = _add_divergence(df)

    # ── Xác định feature columns ──────────────────────────────────────────
    meta_cols = {'timestamp', 'open', 'high', 'low', 'close', 'volume',
                 'vwap', 'ema5', 'ema10', 'ema20', 'ema50',
                 'bb_upper', 'bb_lower', 'bb_width', 'atr',
                 'volume_ma20', 'sr_high', 'sr_low'}
    feature_cols = [c for c in df.columns if c not in meta_cols and df[c].dtype != object]

    if dropna:
        df.dropna(subset=feature_cols, inplace=True)
        df.reset_index(drop=True, inplace=True)

    return df, feature_cols


# ── Helper: merge slope từ higher TF ─────────────────────────────────────────

def _merge_tf_slope(
    df_base: pd.DataFrame,
    df_higher: pd.DataFrame | None,
    col_name: str,
) -> pd.DataFrame:
    if df_higher is None or df_higher.empty:
        df_base[col_name] = 0.0
        return df_base
    tmp = _ensure_utc_ns(df_higher.copy())
    tmp['ema20']    = EMAIndicator(tmp['close'], 20).ema_indicator()
    tmp[col_name]   = tmp['ema20'].pct_change() * 100
    df_base = _ensure_utc_ns(df_base)
    merged = pd.merge_asof(
        df_base.sort_values('timestamp'),
        tmp[['timestamp', col_name]].sort_values('timestamp'),
        on='timestamp', direction='backward',
    )
    return merged
