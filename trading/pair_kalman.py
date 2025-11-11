# trading/pair_kalman.py
from __future__ import annotations

import numpy as np
import pandas as pd

from .pair_trading import PairTradingStrategy

__all__ = ["KalmanPairsStrategy"]


class KalmanPairsStrategy(PairTradingStrategy):
    """
    Pair trading with a time-varying hedge ratio β(t) estimated by a
    2-state Kalman filter on the regression:

        y_t = α_t + β_t * x_t + ε_t

    State model (random walk):
        α_t = α_{t-1} + w_t^α
        β_t = β_{t-1} + w_t^β

    where w_t ~ N(0, Q) with Q = diag(q_alpha, q_beta) and
    observation noise ε_t ~ N(0, r).

    Trading signal:
      - We use the model *innovation* (residual) as the spread and build a
        rolling z-score on a window = `lookback`. Positions:
          z >  z_entry  -> short spread  (position = -1)
          z < -z_entry  -> long  spread  (position = +1)
          |z| < z_exit  -> flat
      - We expose the exact hedge ratio used for PnL in column `hedge_ratio`
        so the backtester can consume the same β(t) (shifted by 1 day).

    Notes:
      * We clip β(t) to +/- cap_beta for robustness.
      * We floor the rolling std to avoid division explosions in z-score.
    """

    def __init__(
        self,
        lookback: int = 60,
        z_entry: float = 2.0,
        z_exit: float = 0.5,
        q_alpha: float = 1e-6,
        q_beta: float = 1e-6,
        r: float = 1e-3,
        init_window: int = 60,
        cap_beta: float = 5.0,
    ):
        super().__init__(lookback=lookback, z_entry=z_entry, z_exit=z_exit)
        self.q_alpha = float(q_alpha)
        self.q_beta = float(q_beta)
        self.r = float(r)
        self.init_window = int(init_window)
        self.cap_beta = float(cap_beta)

    # --------------------- Kalman core ---------------------
    @staticmethod
    def _kalman_filter_xy(
        y: np.ndarray,
        x: np.ndarray,
        q_alpha: float,
        q_beta: float,
        r: float,
        alpha0: float,
        beta0: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run a 2D Kalman filter for the state θ_t = [α_t, β_t].

        Returns:
          alpha_t (np.ndarray), beta_t (np.ndarray), residual_t (np.ndarray)
        """
        n = len(y)
        alpha = np.zeros(n)
        beta = np.zeros(n)
        resid = np.zeros(n)

        # State
        theta = np.array([alpha0, beta0], dtype=float)  # [α, β]
        P = np.eye(2) * 1.0                            # initial covariance
        Q = np.diag([q_alpha, q_beta])                 # process noise
        R = float(r)                                   # obs noise

        for t in range(n):
            # Predict (random walk model)
            theta_pred = theta
            P_pred = P + Q

            # Measurement y_t = [1, x_t] θ + v_t
            H = np.array([[1.0, float(x[t])]])
            y_pred = float(H @ theta_pred.reshape(2, 1))
            innov = float(y[t]) - y_pred
            S = float(H @ P_pred @ H.T) + R
            if S <= 1e-12:
                S = 1e-12  # numerical floor

            # Update
            K = (P_pred @ H.T).reshape(2) / S  # 2x1 -> vector
            theta = theta_pred + K * innov
            P = (np.eye(2) - np.outer(K, H)) @ P_pred

            alpha[t], beta[t] = theta[0], theta[1]
            resid[t] = innov

        return alpha, beta, resid

    # ------------------ Signals API (pairs) ------------------
    def generate_pair_signals(self, df_prices: pd.DataFrame, col1: str, col2: str) -> pd.DataFrame:
        """
        Parameters
        ----------
        df_prices : DataFrame
            Must contain price columns `col1`, `col2`.
        col1, col2 : str
            Names of the two price columns (y = col1, x = col2).

        Returns
        -------
        DataFrame indexed by date with columns:
            alpha, beta, hedge_ratio, spread, zscore, position
        """
        df = df_prices[[col1, col2]].copy().astype(float).dropna()
        if df.empty:
            return pd.DataFrame(
                columns=["alpha", "beta", "hedge_ratio", "spread", "zscore", "position"]
            )

        y = df[col1].values  # dependent
        x = df[col2].values  # regressor

        # Warm-up OLS for initial state (α0, β0)
        w0 = max(self.init_window, 10)
        if len(df) < w0 + 5:
            # Not enough data: fallback to static spread/zscore
            return super().generate_pair_signals(df_prices, col1, col2)

        x0, y0 = x[:w0], y[:w0]
        vx = np.var(x0)
        beta0 = (np.cov(y0, x0, ddof=0)[0, 1] / vx) if vx > 0 else 1.0
        if not np.isfinite(beta0):
            beta0 = 1.0
        alpha0 = float(np.mean(y0 - beta0 * x0))

        # Run Kalman
        alpha, beta, resid = self._kalman_filter_xy(
            y=y,
            x=x,
            q_alpha=self.q_alpha,
            q_beta=self.q_beta,
            r=self.r,
            alpha0=alpha0,
            beta0=beta0,
        )

        # Clip β for robustness (optional)
        if self.cap_beta and self.cap_beta > 0:
            beta = np.clip(beta, -self.cap_beta, self.cap_beta)

        # Innovation is the spread
        spread = pd.Series(resid, index=df.index, name="spread")

        # Rolling z-score on the innovation
        roll_mean = spread.rolling(self.lookback, min_periods=self.lookback).mean()
        roll_std = spread.rolling(self.lookback, min_periods=self.lookback).std()
        # Avoid division by ~0
        roll_std = roll_std.where(roll_std > 1e-8, 1e-8)

        zscore = (spread - roll_mean) / roll_std

        out = pd.DataFrame(index=df.index)
        out["alpha"] = pd.Series(alpha, index=df.index)
        out["beta"] = pd.Series(beta, index=df.index)
        # <- this is what your backtester should use (shifted by 1 day inside the BT)
        out["hedge_ratio"] = out["beta"]

        out["spread"] = spread
        out["zscore"] = zscore

        # Position logic with hysteresis
        out["position"] = 0
        valid = out["zscore"].notna()
        out.loc[valid & (out["zscore"] > self.z_entry), "position"] = -1
        out.loc[valid & (out["zscore"] < -self.z_entry), "position"] = +1
        inside = valid & (out["zscore"].abs() < self.z_exit)
        out.loc[inside, "position"] = 0
        # Carry positions forward but keep NaN warm-up as flat
        out["position"] = out["position"].replace(0, np.nan).ffill().fillna(0).astype(int)

        return out
