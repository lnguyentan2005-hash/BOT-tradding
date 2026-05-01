import lightgbm as lgb
import joblib
import os
import numpy as np

class ModelManager:
    def __init__(self, model_file, lgbm_params, lookback_window):
        self.model_file = model_file
        self.lgbm_params = lgbm_params
        self.lookback_window = lookback_window
        self.model = None
        self.feature_cols = None

    def prepare_data(self, df_merged):
        data = df_merged.copy()
        data['target'] = (data['close'].shift(-1) > data['open'].shift(-1)).astype(int)
        data.dropna(subset=['target'], inplace=True)

        feature_cols = [col for col in data.columns if col != 'target']
        self.feature_cols = feature_cols

        X_list, y_list, ts_list = [], [], []
        for i in range(self.lookback_window, len(data)-1):
            window = data.iloc[i-self.lookback_window:i][feature_cols].values.flatten()
            X_list.append(window)
            y_list.append(data.iloc[i]['target'])
            ts_list.append(data.index[i])
        return np.array(X_list), np.array(y_list), np.array(ts_list)

    def train(self, X_train, y_train, X_valid=None, y_valid=None):
        model = lgb.LGBMClassifier(**self.lgbm_params)
        if X_valid is not None and y_valid is not None:
            model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_valid, y_valid)],
                eval_names=['train', 'valid'],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(10)]
            )
        else:
            model.fit(X_train, y_train)
        self.model = model
        return model

    def save_model(self):
        os.makedirs(os.path.dirname(self.model_file), exist_ok=True)
        joblib.dump(self.model, self.model_file)
        joblib.dump(self.feature_cols, self.model_file.replace('.pkl', '_features.pkl'))
        print(f"💾 Đã lưu mô hình vào {self.model_file}")

    def load_model(self, current_feature_count=None):
        if os.path.exists(self.model_file):
            self.model = joblib.load(self.model_file)
            feat_file = self.model_file.replace('.pkl', '_features.pkl')
            if os.path.exists(feat_file):
                self.feature_cols = joblib.load(feat_file)

                if current_feature_count is not None:
                    saved_count = len(self.feature_cols) * self.lookback_window
                    if saved_count != current_feature_count:
                        print(f"⚠️ Model cũ có {saved_count} đặc trưng, không khớp với hiện tại ({current_feature_count}).")
                        print("   Xóa model cũ và sẽ huấn luyện lại...")
                        os.remove(self.model_file)
                        if os.path.exists(feat_file):
                            os.remove(feat_file)
                        self.model = None
                        self.feature_cols = None
                        return False

            print(f"📦 Đã nạp mô hình từ {self.model_file}")
            return True
        return False

    def predict_proba(self, X):
        if self.model is None:
            raise Exception("Model chưa được nạp hoặc huấn luyện")
        return self.model.predict_proba(X)[:, 1]