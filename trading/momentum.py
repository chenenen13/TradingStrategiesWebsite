# trading/momentum.py
from __future__ import annotations

import pandas as pd
from .strategie_main import BaseStrategy


class MomentumStrategy(BaseStrategy):
    """
    Simple SMA crossover:
      - long when MA_short > MA_long
      - short when MA_short < MA_long (put 0 if you prefer long/cash)
    """
    def __init__(self, short_window: int = 20, long_window: int = 50, allow_short: bool = True):
        super().__init__(name="Momentum")
        self.short_window = int(short_window)
        self.long_window = int(long_window)
        self.allow_short = bool(allow_short)

    def generate_signals(self, price_series) -> pd.DataFrame:
        prices = price_series.data["price"].astype(float)
        df = price_series.data.copy()

        df["ma_short"] = prices.rolling(self.short_window, min_periods=self.short_window).mean()
        df["ma_long"]  = prices.rolling(self.long_window,  min_periods=self.long_window).mean()

        df["signal"] = 0
        mask_valid = df["ma_short"].notna() & df["ma_long"].notna()
        gt = df["ma_short"] > df["ma_long"]
        lt = df["ma_short"] < df["ma_long"]

        df.loc[mask_valid & gt, "signal"] = 1
        if self.allow_short:
            df.loc[mask_valid & lt, "signal"] = -1   # set to 0 for long-only

        # drop warm-up rows where moving averages are NaN
        return df.dropna(subset=["ma_short", "ma_long"])
