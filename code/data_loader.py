import os
import pandas as pd
import requests
import time
from datetime import datetime

def _get_cache_filename(symbol, cache_dir, interval, start_str, end_str):
    start_clean = start_str.replace(' ', '_').replace(':', '-')
    end_clean = end_str.replace(' ', '_').replace(':', '-')
    return os.path.join(cache_dir, f"{symbol}_{interval}_{start_clean}_{end_clean}.csv")

def _get_merged_cache_filename(symbol, cache_dir, start_str, end_str):
    start_clean = start_str.replace(' ', '_').replace(':', '-')
    end_clean = end_str.replace(' ', '_').replace(':', '-')
    return os.path.join(cache_dir, f"{symbol}_MERGED_INDICATORS_{start_clean}_{end_clean}.csv")

def fetch_klines(symbol, cache_dir, interval, start_date, end_date, force_download=False, verbose=True):
    cache_file = _get_cache_filename(symbol, cache_dir, interval, start_date, end_date)
    if not force_download and os.path.exists(cache_file):
        if verbose:
            print(f"📂 Load {interval} từ cache")
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index)
        return df

    base_url = "https://api.binance.com/api/v3/klines"
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d %H:%M").timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d %H:%M").timestamp() * 1000)

    all_data = []
    current_start = start_ts
    if verbose:
        print(f"⏳ Tải {interval}...")

    max_retries = 5
    for attempt in range(max_retries):
        try:
            while current_start < end_ts:
                params = {
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": current_start,
                    "limit": 1000
                }
                resp = requests.get(base_url, params=params, timeout=30)
                data = resp.json()
                if not isinstance(data, list) or len(data) == 0:
                    break
                all_data.extend(data)
                current_start = data[-1][6] + 1
                if len(data) < 1000:
                    break
                time.sleep(0.3)
            break  # thành công
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            print(f"⚠️ Lỗi mạng (lần {attempt+1}/{max_retries}): {e}")
            time.sleep(5)
    else:
        raise ValueError(f"Không thể kết nối Binance sau {max_retries} lần thử.")

    if not all_data:
        raise ValueError(f"Không tải được dữ liệu {interval}!")

    columns = ["open_time", "open", "high", "low", "close", "volume",
               "close_time", "quote_vol", "trades", "taker_buy_base",
               "taker_buy_quote", "ignore"]
    df = pd.DataFrame(all_data, columns=columns)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col])
    df.set_index("open_time", inplace=True)
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    df = df.sort_index().drop_duplicates()

    os.makedirs(cache_dir, exist_ok=True)
    df.to_csv(cache_file)
    if verbose:
        print(f"💾 Đã lưu cache ({len(df)} nến)")
    return df

def get_all_data(symbol, cache_dir, intervals, start_date, end_date, force_download=False):
    data = {}
    for interval in intervals:
        data[interval] = fetch_klines(symbol, cache_dir, interval, start_date, end_date, force_download)
    return data

def load_merged_data(symbol, cache_dir, start_str, end_str):
    cache_file = _get_merged_cache_filename(symbol, cache_dir, start_str, end_str)
    if os.path.exists(cache_file):
        print(f"📂 Load dữ liệu merge từ cache")
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index)
        return df
    return None

def save_merged_data(df, symbol, cache_dir, start_str, end_str):
    cache_file = _get_merged_cache_filename(symbol, cache_dir, start_str, end_str)
    os.makedirs(cache_dir, exist_ok=True)
    df.to_csv(cache_file)
    print(f"💾 Đã lưu cache merge")