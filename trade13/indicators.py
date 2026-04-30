# === FILE: indicators.py ===
"""
indicators.py – Backward-compatible wrapper.
Code cũ gọi compute_indicators() vẫn hoạt động qua generate_features().
"""
import pandas as pd
from features import generate_features


def compute_indicators(
    df_5m: pd.DataFrame,
    df_15m: pd.DataFrame | None = None,
    df_30m: pd.DataFrame | None = None,
    df_1h: pd.DataFrame | None = None,
    dropna: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Wrapper: gọi generate_features() với cùng interface."""
    return generate_features(df_5m, df_15m, df_30m, df_1h, dropna=dropna)
