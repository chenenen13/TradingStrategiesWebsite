# trading/outliers.py
import pandas as pd

from .models import PriceSeries

class OutlierDetector:
    def __init__(self, window: int = 20, z_threshold: float = 3.0):
        self.window = window
        self.z_threshold = z_threshold

    def detect(self, price_series: PriceSeries) -> pd.DataFrame:
        prices = price_series.data['price']
        df = price_series.data.copy()
        df['returns'] = prices.pct_change()

        rolling_mean = df['returns'].rolling(self.window).mean()
        rolling_std = df['returns'].rolling(self.window).std()

        df['zscore'] = (df['returns'] - rolling_mean) / rolling_std
        df['outlier'] = df['zscore'].abs() > self.z_threshold

        return df
