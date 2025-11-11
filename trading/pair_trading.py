# trading/pair_trading.py
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from .strategie_main import BaseStrategy


class PairTradingStrategy(BaseStrategy):
    """
    Static beta spread with rolling z-score.
    Trading logic:
      - short spread when z > z_entry  (position = -1)
      - long  spread when z < -z_entry (position = +1)
      - flat when |z| < z_exit
    """
    def __init__(self, lookback: int = 60, z_entry: float = 2.0, z_exit: float = 0.5):
        super().__init__(name="PairTrading")
        self.lookback = int(lookback)
        self.z_entry = float(z_entry)
        self.z_exit = float(z_exit)

    # ---------- helpers ----------
    @staticmethod
    def _ols_beta(s1: pd.Series, s2: pd.Series) -> float:
        """Return OLS beta of s1 on s2."""
        x = s2.astype(float).values
        y = s1.astype(float).values
        vx = np.var(x)
        if vx == 0 or np.isnan(vx):
            return 1.0
        cov = np.cov(y, x, ddof=0)[0, 1]
        beta = cov / vx
        if np.isnan(beta) or np.isinf(beta):
            beta = 1.0
        return float(beta)

    def test_cointegration(self, s1: pd.Series, s2: pd.Series):
        """Return (stat, p-value, crit_values) using Engle–Granger."""
        s1, s2 = s1.align(s2, join="inner")
        s1, s2 = s1.dropna(), s2.dropna()
        s1, s2 = s1.align(s2, join="inner")
        if len(s1) < 30:
            return np.nan, np.nan, {}
        coint_t, p_value, crit = coint(s1.values, s2.values)
        return coint_t, p_value, crit

    def _build_spread(self, s1: pd.Series, s2: pd.Series) -> pd.Series:
        beta = self._ols_beta(s1, s2)
        return s1 - beta * s2

    # ---------- API ----------
    def generate_signals(self, _price_series) -> pd.DataFrame:
        """Pairs strategy is defined on two series, use generate_pair_signals()."""
        raise NotImplementedError("Use generate_pair_signals(df_prices, col1, col2).")

    def generate_pair_signals(self, df_prices: pd.DataFrame, col1: str, col2: str) -> pd.DataFrame:
        """
        Parameters
        ----------
        df_prices : DataFrame with columns [col1, col2]
        col1, col2: column names

        Returns
        -------
        DataFrame with:
          - spread
          - zscore
          - position in {+1, 0, -1}
        """
        df = df_prices[[col1, col2]].copy().astype(float).dropna()
        s1 = df[col1]
        s2 = df[col2]

        spread = self._build_spread(s1, s2)
        roll_mean = spread.rolling(self.lookback, min_periods=self.lookback).mean()
        roll_std  = spread.rolling(self.lookback, min_periods=self.lookback).std()
        zscore = (spread - roll_mean) / roll_std

        out = pd.DataFrame(index=df.index)
        out["spread"] = spread
        out["zscore"] = zscore

        out["position"] = 0
        valid = out["zscore"].notna()
        out.loc[valid & (out["zscore"] >  self.z_entry), "position"] = -1
        out.loc[valid & (out["zscore"] < -self.z_entry), "position"] = +1

        inside = valid & (out["zscore"].abs() < self.z_exit)
        out.loc[inside, "position"] = 0

        # carry position across time, keep NaN periods flat
        out["position"] = out["position"].replace(0, np.nan).ffill().fillna(0).astype(int)
        return out
