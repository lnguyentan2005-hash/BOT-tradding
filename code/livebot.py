import os
import sys
import time
import warnings
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import pytz
import requests
from colorama import init, Fore, Style

from config_live import (
    SYMBOL, INTERVAL, EXTRA_TIMEFRAMES, LOOKBACK_WINDOW,
    STRONG_PROB_THRESHOLD, USE_TREND_FILTER, EMA_TREND_PERIOD,
    CACHE_DIR, MODEL_FILE, LGBM_PARAMS
)
from data_loader import fetch_klines
from indicators import compute_indicators, merge_with_higher_tf
from model_manager import ModelManager

warnings.filterwarnings('ignore', message='X does not have valid feature names')
warnings.filterwarnings('ignore', category=DeprecationWarning)
init(autoreset=True)

VN_TZ = pytz.timezone('Asia/Ho_Chi_Minh')
EXCEL_LOG = "live_trade_log.xlsx"
PREDICT_BEFORE_SEC = 1

class LiveBot:
    def __init__(self):
        print(f"{Fore.CYAN}Initializing LiveBot...")
        self.model_mgr = ModelManager(MODEL_FILE, LGBM_PARAMS, LOOKBACK_WINDOW)
        if not self.model_mgr.load_model():
            raise FileNotFoundError(f"❌ Không tìm thấy mô hình {MODEL_FILE}")
        self.raw_cache = {}
        self.indicator_cache = {}
        if self.model_mgr.feature_cols is None:
            print(f"{Fore.RED}❌ Model không có feature_cols. Hãy train lại model.")
            sys.exit(1)
        self.feature_cols = self.model_mgr.feature_cols
        print(f"{Fore.GREEN}✅ Đã nạp {len(self.feature_cols)} đặc trưng từ model")
        self.expected_features = len(self.feature_cols) * LOOKBACK_WINDOW
        print(f"{Fore.GREEN}✅ Model kỳ vọng {self.expected_features} đặc trưng đầu vào")

    def get_current_price(self):
        url = f"https://api.binance.com/api/v3/ticker/price?symbol={SYMBOL}"
        try:
            resp = requests.get(url, timeout=3)
            return float(resp.json()['price'])
        except Exception as e:
            print(f"{Fore.YELLOW}⚠️ Không lấy được giá hiện tại: {e}")
            return None

    def _deduplicate_and_sort(self, df):
        if df is None or df.empty:
            return df
        df = df[~df.index.duplicated(keep='last')]
        df = df.sort_index()
        return df

    def fetch_missing_data(self, interval, now_utc_naive):
        # Giữ nguyên code cũ (dùng UTC) - OK
        try:
            if interval not in self.raw_cache or self.raw_cache[interval].empty:
                if interval.endswith('m'):
                    mins = int(interval[:-1])
                    start = now_utc_naive - timedelta(minutes=600 * mins)
                elif interval.endswith('h'):
                    hrs = int(interval[:-1])
                    start = now_utc_naive - timedelta(hours=600 * hrs)
                elif interval.endswith('d'):
                    days = int(interval[:-1])
                    start = now_utc_naive - timedelta(days=600 * days)
                else:
                    start = now_utc_naive - timedelta(days=10)
                if start > now_utc_naive:
                    start = now_utc_naive - timedelta(days=1)
                start_str = start.strftime("%Y-%m-%d %H:%M")
                end_str = now_utc_naive.strftime("%Y-%m-%d %H:%M")
                df = fetch_klines(SYMBOL, CACHE_DIR, interval, start_str, end_str,
                                  force_download=True, verbose=False)
                if df.empty:
                    raise ValueError(f"Không có dữ liệu {interval}")
                df = self._deduplicate_and_sort(df)
                self.raw_cache[interval] = df
            else:
                last_ts = self.raw_cache[interval].index.max()
                start = last_ts + timedelta(minutes=1)
                if start > now_utc_naive:
                    return
                start_str = start.strftime("%Y-%m-%d %H:%M")
                end_str = now_utc_naive.strftime("%Y-%m-%d %H:%M")
                new_df = fetch_klines(SYMBOL, CACHE_DIR, interval, start_str, end_str,
                                      force_download=True, verbose=False)
                if not new_df.empty:
                    new_df = self._deduplicate_and_sort(new_df)
                    combined = pd.concat([self.raw_cache[interval], new_df])
                    combined = self._deduplicate_and_sort(combined)
                    self.raw_cache[interval] = combined.iloc[-600:]
                else:
                    self.raw_cache[interval] = self._deduplicate_and_sort(self.raw_cache[interval])
        except Exception as e:
            print(f"{Fore.RED}❌ Lỗi tải {interval}: {e}")
            if interval not in self.raw_cache:
                self.raw_cache[interval] = pd.DataFrame()
            raise

    def update_all_data(self):
        now_utc_naive = datetime.now(timezone.utc).replace(tzinfo=None)
        intervals = [INTERVAL] + EXTRA_TIMEFRAMES
        with ThreadPoolExecutor(max_workers=len(intervals)) as executor:
            futures = {executor.submit(self.fetch_missing_data, iv, now_utc_naive): iv for iv in intervals}
            for future in as_completed(futures):
                iv = futures[future]
                try:
                    future.result()
                except Exception:
                    print(f"{Fore.YELLOW}⚠️ Bỏ qua khung {iv} do lỗi, tiếp tục")
                    continue

    def compute_all_indicators(self):
        intervals = [INTERVAL] + EXTRA_TIMEFRAMES
        def compute(interval):
            df_raw = self.raw_cache.get(interval)
            if df_raw is None or df_raw.empty:
                return interval, pd.DataFrame()
            df_raw = self._deduplicate_and_sort(df_raw)
            try:
                df_ind = compute_indicators(df_raw)
                return interval, df_ind
            except Exception as e:
                print(f"{Fore.RED}❌ Lỗi tính chỉ báo {interval}: {e}")
                return interval, pd.DataFrame()
        with ThreadPoolExecutor(max_workers=len(intervals)) as executor:
            futures = {executor.submit(compute, iv): iv for iv in intervals}
            for future in as_completed(futures):
                iv, df_ind = future.result()
                if not df_ind.empty:
                    self.indicator_cache[iv] = df_ind
                else:
                    print(f"{Fore.YELLOW}⚠️ Không có chỉ báo cho {iv}")

    def predict_signal(self, candle_start_vn, candle_close_vn):
        try:
            self.update_all_data()
            if INTERVAL not in self.raw_cache or self.raw_cache[INTERVAL].empty:
                return "SKIP", 0.5, f"Không có dữ liệu {INTERVAL}", None, None, None, None
            self.compute_all_indicators()
            if INTERVAL not in self.indicator_cache or self.indicator_cache[INTERVAL].empty:
                return "SKIP", 0.5, f"Không có chỉ báo {INTERVAL}", None, None, None, None

            df_5m_ind = self.indicator_cache[INTERVAL]
            dict_htf_ind = {}
            for tf in EXTRA_TIMEFRAMES:
                if tf in self.indicator_cache and not self.indicator_cache[tf].empty:
                    dict_htf_ind[tf] = self.indicator_cache[tf]
            df_merged = merge_with_higher_tf(df_5m_ind, dict_htf_ind)
            if df_merged.empty or len(df_merged) < LOOKBACK_WINDOW:
                return "SKIP", 0.5, f"Không đủ dữ liệu merged (cần {LOOKBACK_WINDOW} nến)", None, None, None, None

            missing_cols = [c for c in self.feature_cols if c not in df_merged.columns]
            if missing_cols:
                return "SKIP", 0.5, f"Thiếu cột trong merged: {missing_cols[:5]}...", None, None, None, None

            # ---- CHUYỂN ĐỔI MÚI GIỜ CHÍNH XÁC ----
            # candle_start_vn là datetime có timezone VN
            candle_start_utc = candle_start_vn.astimezone(timezone.utc).replace(tzinfo=None)
            print(f"   Debug: nến VN start {candle_start_vn} -> UTC {candle_start_utc}")

            # Lấy quá khứ: tất cả các nến có index < candle_start_utc
            past = df_merged[df_merged.index < candle_start_utc].copy()
            if len(past) < LOOKBACK_WINDOW - 1:
                return "SKIP", 0.5, f"Không đủ {LOOKBACK_WINDOW-1} nến quá khứ (có {len(past)})", None, None, None, None
            past = past.iloc[-(LOOKBACK_WINDOW - 1):]

            current_price = self.get_current_price()
            if current_price is None:
                current_price = df_merged.iloc[-1]['close']

            # Tạo row hiện tại với index = candle_start_utc
            current_row = pd.DataFrame(index=[candle_start_utc], columns=df_merged.columns)
            # Copy giá trị từ nến trước đó (giữ nguyên chỉ báo)
            for col in df_merged.columns:
                current_row[col] = past.iloc[-1][col]
            # Gán giá realtime
            current_row['open'] = current_price
            current_row['high'] = current_price
            current_row['low'] = current_price
            current_row['close'] = current_price

            extended = pd.concat([past, current_row])
            extended = self._deduplicate_and_sort(extended)
            recent = extended.iloc[-LOOKBACK_WINDOW:]
            X = recent[self.feature_cols].values.flatten().reshape(1, -1)
            if X.shape[1] != self.expected_features:
                return "SKIP", 0.5, f"Số lượng đặc trưng {X.shape[1]} != {self.expected_features}", None, None, None, None

            prob_up = self.model_mgr.predict_proba(X)[0]

            signal = "SKIP"
            reason = ""
            ema_col = f'ema{EMA_TREND_PERIOD}'
            use_filter = USE_TREND_FILTER and (ema_col in df_merged.columns)
            # Lấy nến đã đóng gần nhất để so sánh EMA
            last_closed = past.iloc[-1] if len(past) > 0 else None
            if prob_up >= STRONG_PROB_THRESHOLD:
                if use_filter and last_closed is not None:
                    ema_val = last_closed[ema_col]
                    if current_price > ema_val:
                        signal = "UP"
                        reason = f"UP {prob_up:.3f} >= {STRONG_PROB_THRESHOLD}, price>{ema_col}={ema_val:.2f}"
                    else:
                        reason = f"UP {prob_up:.3f} nhưng price <= {ema_col} ({ema_val:.2f})"
                else:
                    signal = "UP"
                    reason = f"UP {prob_up:.3f} >= {STRONG_PROB_THRESHOLD}"
            elif prob_up <= 1 - STRONG_PROB_THRESHOLD:
                if use_filter and last_closed is not None:
                    ema_val = last_closed[ema_col]
                    if current_price < ema_val:
                        signal = "DOWN"
                        reason = f"DOWN {1-prob_up:.3f} >= {STRONG_PROB_THRESHOLD}, price<{ema_col}={ema_val:.2f}"
                    else:
                        reason = f"DOWN {1-prob_up:.3f} nhưng price >= {ema_col} ({ema_val:.2f})"
                else:
                    signal = "DOWN"
                    reason = f"DOWN {1-prob_up:.3f} >= {STRONG_PROB_THRESHOLD}"
            else:
                reason = f"UP={prob_up:.3f} trung tính"

            return signal, prob_up, reason, current_price, None, current_price, current_price
        except Exception as e:
            print(f"{Fore.RED}Lỗi predict_signal: {e}")
            import traceback
            traceback.print_exc()
            return "SKIP", 0.5, f"Lỗi: {e}", None, None, None, None

    def log_prediction(self, candle_start_vn, signal, open_price, high_price, low_price, close_price, reason, process_time_sec):
        try:
            # Lưu timestamp theo VN (chỉ để hiển thị)
            ts_vn = candle_start_vn
            if ts_vn.tzinfo is None:
                ts_vn = VN_TZ.localize(ts_vn)
            else:
                ts_vn = ts_vn.astimezone(VN_TZ).replace(tzinfo=None)
            new_row = {
                'timestamp': ts_vn,
                'open': round(open_price, 2) if open_price else None,
                'high': round(high_price, 2) if high_price else None,
                'low': round(low_price, 2) if low_price else None,
                'close': None,
                'signal': signal,
                'winlose': 'PENDING',
                'reason': reason,
                'process_time_sec': round(process_time_sec, 3)
            }
            if not os.path.exists(EXCEL_LOG):
                log_df = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'signal', 'winlose', 'reason', 'process_time_sec'])
            else:
                log_df = pd.read_excel(EXCEL_LOG, engine='openpyxl')
            log_df = pd.concat([log_df, pd.DataFrame([new_row])], ignore_index=True)
            log_df.to_excel(EXCEL_LOG, index=False, engine='openpyxl')
            print(f"   📝 Đã ghi log: {signal} lúc {ts_vn}")
        except Exception as e:
            print(f"⚠️ Lỗi ghi Excel: {e}")

    def update_pending(self, df_merged):
        try:
            if not os.path.exists(EXCEL_LOG):
                return
            log_df = pd.read_excel(EXCEL_LOG, engine='openpyxl')
            pending = log_df[log_df['winlose'] == 'PENDING']
            for idx, row in pending.iterrows():
                # timestamp lưu trong log là VN (naive)
                ts_vn = pd.Timestamp(row['timestamp'])
                if ts_vn.tzinfo is None:
                    ts_vn = VN_TZ.localize(ts_vn)
                ts_utc = ts_vn.astimezone(timezone.utc).replace(tzinfo=None)
                # Cập nhật nến đã đóng
                if ts_utc in df_merged.index:
                    actual = df_merged.loc[ts_utc]
                    log_df.at[idx, 'open'] = round(actual['open'], 2)
                    log_df.at[idx, 'high'] = round(actual['high'], 2)
                    log_df.at[idx, 'low'] = round(actual['low'], 2)
                    log_df.at[idx, 'close'] = round(actual['close'], 2)
                else:
                    continue
                # Xác định nến tiếp theo
                next_ts_utc = ts_utc + pd.Timedelta(minutes=5)
                if next_ts_utc in df_merged.index:
                    next_row = df_merged.loc[next_ts_utc]
                    actual_up = next_row['close'] > next_row['open']
                    if row['signal'] == 'UP':
                        log_df.at[idx, 'winlose'] = 'WIN' if actual_up else 'LOSE'
                    elif row['signal'] == 'DOWN':
                        log_df.at[idx, 'winlose'] = 'WIN' if not actual_up else 'LOSE'
                    else:
                        log_df.at[idx, 'winlose'] = 'SKIP'
                    result = log_df.at[idx, 'winlose']
                    color = Fore.GREEN if result == 'WIN' else Fore.RED
                    icon = "✅" if result == 'WIN' else "❌"
                    print(f"{color}{icon} KẾT QUẢ LỆNH {row['signal']} lúc {row['timestamp']}: {result}")
            log_df.to_excel(EXCEL_LOG, index=False, engine='openpyxl')
        except Exception as e:
            print(f"⚠️ Lỗi cập nhật Excel: {e}")

    def run(self):
        print(f"{Fore.MAGENTA}{'='*60}")
        print(f"{Fore.YELLOW}   BTC 5M LIVE PREDICTOR (Dự đoán ở giây thứ 59)")
        print(f"{Fore.MAGENTA}{'='*60}\n")

        now_vn = datetime.now(VN_TZ)
        minutes = now_vn.minute
        remainder = minutes % 5
        delta_to_next = (5 - remainder) * 60 - now_vn.second - (now_vn.microsecond / 1_000_000)
        if delta_to_next <= 0:
            delta_to_next += 300
        next_candle_close = now_vn + timedelta(seconds=delta_to_next)
        next_candle_close = next_candle_close.replace(microsecond=0)

        print(f"{Fore.WHITE}⏳ Nến hiện tại đóng lúc: {next_candle_close.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{Fore.WHITE}⏳ Dự đoán khi còn {PREDICT_BEFORE_SEC} giây (giây thứ 59)\n")

        while True:
            try:
                now_vn = datetime.now(VN_TZ)
                predict_time = next_candle_close - timedelta(seconds=PREDICT_BEFORE_SEC)

                if now_vn < predict_time:
                    remaining = (predict_time - now_vn).total_seconds()
                    if remaining <= 55:
                        self.countdown(remaining, f"⏰ Còn {int(remaining)}s đến dự đoán")
                    else:
                        mins, secs = divmod(int(remaining), 60)
                        sys.stdout.write(f"\r{Fore.CYAN}⏳ Chờ đến giây 59: {mins:02d}:{secs:02d}   ")
                        sys.stdout.flush()
                        time.sleep(0.5)
                    continue

                print(f"\n{Fore.YELLOW}{'='*60}")
                print(f"{Fore.WHITE}📡 THỰC HIỆN DỰ ĐOÁN TẠI {now_vn.strftime('%Y-%m-%d %H:%M:%S')} (giây 59)")

                candle_close_vn = next_candle_close
                candle_start_vn = candle_close_vn - timedelta(minutes=5)
                # Đảm bảo candle_start_vn có timezone VN
                if candle_start_vn.tzinfo is None:
                    candle_start_vn = VN_TZ.localize(candle_start_vn)
                if candle_close_vn.tzinfo is None:
                    candle_close_vn = VN_TZ.localize(candle_close_vn)

                t_start = time.perf_counter()
                print(f"   📥 Đang tải và xử lý dữ liệu...")
                signal, prob_up, reason, open_price, close_price, high_price, low_price = self.predict_signal(candle_start_vn, candle_close_vn)
                elapsed = time.perf_counter() - t_start
                print(f"   ✅ Hoàn tất sau {elapsed:.3f} giây")

                if signal == "UP":
                    print(f"{Fore.GREEN}   ⇒ TÍN HIỆU: {signal} (Xác suất UP = {prob_up:.3f})")
                elif signal == "DOWN":
                    print(f"{Fore.RED}   ⇒ TÍN HIỆU: {signal} (Xác suất DOWN = {1-prob_up:.3f})")
                else:
                    print(f"{Fore.YELLOW}   ⇒ TÍN HIỆU: {signal} (UP={prob_up:.3f})")
                print(f"   📝 Lý do: {reason}")
                print(f"   💰 Giá hiện tại: {open_price if open_price else 'N/A'}")

                self.log_prediction(candle_start_vn, signal, open_price, high_price, low_price, close_price, reason, elapsed)

                # Cập nhật pending
                self.update_all_data()
                self.compute_all_indicators()
                df_5m_ind = self.indicator_cache.get(INTERVAL, pd.DataFrame())
                dict_htf_ind = {tf: self.indicator_cache.get(tf, pd.DataFrame()) for tf in EXTRA_TIMEFRAMES}
                df_merged = merge_with_higher_tf(df_5m_ind, dict_htf_ind)
                self.update_pending(df_merged)

                next_candle_close = candle_close_vn + timedelta(minutes=5)
                print(f"{Fore.YELLOW}{'='*60}\n")
                print(f"{Fore.WHITE}⏳ Nến tiếp theo đóng lúc: {next_candle_close.strftime('%H:%M:%S')}\n")
                time.sleep(0.5)

            except KeyboardInterrupt:
                print(f"\n{Fore.YELLOW}🛑 Bot dừng bởi người dùng.")
                break
            except Exception as e:
                print(f"{Fore.RED}❌ Lỗi nghiêm trọng: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(5)

    def countdown(self, seconds, prefix=""):
        for remaining in range(int(seconds), -1, -1):
            mins, secs = divmod(remaining, 60)
            sys.stdout.write(f"\r{Fore.CYAN}{prefix}: {mins:02d}:{secs:02d}   ")
            sys.stdout.flush()
            if remaining > 0:
                time.sleep(1)
        sys.stdout.write("\r" + " " * 80 + "\r")

if __name__ == "__main__":
    try:
        bot = LiveBot()
        bot.run()
    except Exception as e:
        print(f"Khởi động thất bại: {e}")
        import traceback
        traceback.print_exc()