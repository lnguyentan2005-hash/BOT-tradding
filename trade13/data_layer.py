# === FILE: data_layer.py ===
"""
data_layer.py – Tải OHLCV từ Binance (ccxt), đa khung thời gian, cache CSV.
Hỗ trợ: fetch live, fetch range, fetch history với incremental update.
"""
import time
import os
from datetime import datetime, timedelta, timezone

import ccxt
import pandas as pd
import numpy as np
import pytz

from config import (
    SYMBOL,
    TIMEFRAME_PRIMARY,
    TRAIN_DATA_FILE_5M, TRAIN_DATA_FILE_15M,
    TRAIN_DATA_FILE_30M, TRAIN_DATA_FILE_1H,
)

_CACHE_MAP = {
    '5m':  TRAIN_DATA_FILE_5M,
    '15m': TRAIN_DATA_FILE_15M,
    '30m': TRAIN_DATA_FILE_30M,
    '1h':  TRAIN_DATA_FILE_1H,
}

_TF_MINUTES = {'1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30, '1h': 60, '4h': 240}


class DataLayer:
    """Lớp dữ liệu duy nhất của hệ thống.
    
    - Mọi timestamp đều UTC internally.
    - Các method fetch_* không bao giờ raise exception — log và trả DataFrame rỗng.
    """

    def __init__(self) -> None:
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'},
        })
        self._rate_delay = self.exchange.rateLimit / 1000  # seconds

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _raw_to_df(raw: list) -> pd.DataFrame:
        if not raw:
            return pd.DataFrame()
        df = pd.DataFrame(raw, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.drop_duplicates('timestamp', inplace=True)
        df.sort_values('timestamp', inplace=True)
        df.reset_index(drop=True, inplace=True)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    @staticmethod
    def _normalize_ts(df: pd.DataFrame) -> pd.DataFrame:
        """Đảm bảo timestamp là datetime64[ns, UTC]."""
        if df.empty:
            return df
        df = df.copy()
        ts = pd.to_datetime(df['timestamp'], utc=True)
        df['timestamp'] = ts.astype('datetime64[ns, UTC]')
        return df

    def _add_vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        """Thêm VWAP rolling 20 và khoảng cách giá so với VWAP."""
        if df.empty or len(df) < 5:
            return df
        df = df.copy()
        typical = (df['high'] + df['low'] + df['close']) / 3
        cum_tp_vol = (typical * df['volume']).rolling(20, min_periods=1).sum()
        cum_vol    = df['volume'].rolling(20, min_periods=1).sum()
        df['vwap']      = cum_tp_vol / (cum_vol + 1e-10)
        df['vwap_dist'] = (df['close'] - df['vwap']) / (df['vwap'] + 1e-10) * 100
        return df

    # ── Live fetch ────────────────────────────────────────────────────────────

    def fetch_ohlcv(self, timeframe: str = TIMEFRAME_PRIMARY, limit: int = 300) -> pd.DataFrame:
        """Lấy nến live gần nhất. An toàn — trả empty DF nếu lỗi."""
        try:
            raw = self.exchange.fetch_ohlcv(SYMBOL, timeframe, limit=limit)
            df  = self._raw_to_df(raw)
            df  = self._add_vwap(df)
            return self._normalize_ts(df)
        except Exception as exc:
            print(f"[DataLayer] fetch_ohlcv error ({timeframe}): {exc}")
            return pd.DataFrame()

    # ── Range fetch ───────────────────────────────────────────────────────────

    def fetch_ohlcv_range(
        self,
        timeframe: str,
        since_dt: datetime,
        until_dt: datetime,
        show_progress: bool = False,
    ) -> pd.DataFrame:
        """Tải toàn bộ nến trong [since_dt, until_dt] (UTC) với nhiều request."""
        since_ms = int(since_dt.timestamp() * 1000)
        end_ms   = int(until_dt.timestamp() * 1000)
        tf_ms    = _TF_MINUTES.get(timeframe, 5) * 60 * 1000
        expected = (end_ms - since_ms) // tf_ms
        all_candles: list = []
        current = since_ms

        while current < end_ms:
            try:
                data = self.exchange.fetch_ohlcv(SYMBOL, timeframe, since=current, limit=1000)
                if not data:
                    break
                all_candles.extend(data)
                current = data[-1][0] + tf_ms
                if show_progress and expected > 0:
                    pct = min(len(all_candles) / expected * 100, 100)
                    print(f"\r  Downloading {timeframe}: {len(all_candles)}/{expected} ({pct:.0f}%)  ", end='')
                time.sleep(self._rate_delay)
            except ccxt.RateLimitExceeded:
                time.sleep(30)
            except Exception as exc:
                print(f"\n[DataLayer] range fetch error: {exc}")
                time.sleep(10)

        if show_progress:
            print()

        df = self._raw_to_df(all_candles)
        if df.empty:
            return df
        since_ts = pd.Timestamp(since_dt)
        until_ts = pd.Timestamp(until_dt)
        if since_ts.tzinfo is None:
            since_ts = since_ts.tz_localize('UTC')
        if until_ts.tzinfo is None:
            until_ts = until_ts.tz_localize('UTC')
        mask = (df['timestamp'] >= since_ts) & (df['timestamp'] <= until_ts)
        df = df[mask].reset_index(drop=True)
        df = self._add_vwap(df)
        return self._normalize_ts(df)

    # ── History with incremental cache ────────────────────────────────────────

    def fetch_history(self, timeframe: str, days: int = 180) -> pd.DataFrame:
        """
        Tải `days` ngày lịch sử. 
        - Nếu cache tồn tại và đủ gần: load cache rồi chỉ download phần còn thiếu.
        - Nếu cache quá cũ hoặc không tồn tại: tải lại từ đầu.
        """
        cache_path = _CACHE_MAP.get(timeframe)
        now_utc    = datetime.now(timezone.utc)
        since_dt   = now_utc - timedelta(days=days)

        cached_df = self._load_cache(cache_path)

        if not cached_df.empty:
            last_ts = cached_df['timestamp'].max()
            gap_hours = (now_utc - last_ts.to_pydatetime()).total_seconds() / 3600

            if gap_hours < 6:
                # Cache đủ mới, dùng luôn
                return self._normalize_ts(cached_df)

            # Incremental: chỉ tải phần mới
            print(f"  [{timeframe}] Incremental update: last={last_ts.date()} gap={gap_hours:.1f}h")
            new_df = self.fetch_ohlcv_range(timeframe, last_ts.to_pydatetime(), now_utc, show_progress=True)
            if not new_df.empty:
                merged = pd.concat([cached_df, new_df], ignore_index=True)
                merged.drop_duplicates('timestamp', inplace=True)
                merged.sort_values('timestamp', inplace=True)
                # Giữ đúng khoảng days
                cutoff = pd.Timestamp(since_dt)
                if cutoff.tzinfo is None:
                    cutoff = cutoff.tz_localize('UTC')
                merged = merged[merged['timestamp'] >= cutoff].reset_index(drop=True)
                self._save_cache(merged, cache_path)
                return self._normalize_ts(merged)

        # Full download
        print(f"  [{timeframe}] Full download: {days} days...")
        df = self.fetch_ohlcv_range(timeframe, since_dt, now_utc, show_progress=True)
        self._save_cache(df, cache_path)
        return self._normalize_ts(df)

    # ── Cache I/O ─────────────────────────────────────────────────────────────

    @staticmethod
    def _load_cache(path: str | None) -> pd.DataFrame:
        if not path or not os.path.exists(path):
            return pd.DataFrame()
        try:
            df = pd.read_csv(path, parse_dates=['timestamp'])
            if df.empty:
                return pd.DataFrame()
            if df['timestamp'].dt.tz is None:
                df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
            else:
                df['timestamp'] = df['timestamp'].dt.tz_convert('UTC')
            return df
        except Exception as exc:
            print(f"[DataLayer] cache load error: {exc}")
            return pd.DataFrame()

    @staticmethod
    def _save_cache(df: pd.DataFrame, path: str | None) -> None:
        if not path or df.empty:
            return
        try:
            df.to_csv(path, index=False)
        except Exception as exc:
            print(f"[DataLayer] cache save error: {exc}")
