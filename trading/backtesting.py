# trading/backtesting.py
from __future__ import annotations

import numpy as np
import pandas as pd

from .models import PriceSeries, BacktestResult, PairBacktestResult
from .metrics import PerformanceMetrics
from .strategie_main import BaseStrategy, PairTradingStrategy


class Backtester:
    """
    Vectorized backtester for:
      • Single-asset strategies (signals on one price series)
      • Pair-trading strategies (static β or time-varying β from Kalman)

    Conventions:
      • All P&L is computed on RETURNS (pct_change).
      • Signals and hedge ratios are LAGGED by one bar.
      • Simple transaction costs (bps) are applied on position changes.
    """

    def __init__(
        self,
        initial_capital: float = 10_000.0,
        periods_per_year: int = 252,
        cost_bps_single: float = 1.0,
        cost_bps_pair: float = 1.0,
        max_leverage_beta: float = 10.0,
    ):
        self.initial_capital = float(initial_capital)
        self.ppy = int(periods_per_year)
        self.cost_bps_single = float(cost_bps_single)
        self.cost_bps_pair = float(cost_bps_pair)
        self.max_leverage_beta = float(max_leverage_beta)

    # ------------------------------------------------------------------
    # Single-asset
    # ------------------------------------------------------------------
    def run_single_asset(self, strategy: BaseStrategy, price_series: PriceSeries) -> BacktestResult:
        # 1) Signals
        df = strategy.generate_signals(price_series).copy()

        # Ensure 'price' is present
        if "price" not in df.columns:
            df = df.join(price_series.data[["price"]], how="left")

        # 2) Returns & lagged position
        df["price"] = df["price"].astype(float)
        df["returns"] = df["price"].pct_change().fillna(0.0)

        # lag signal one bar to avoid look-ahead
        df["position"] = df["signal"].astype(float).clip(-1, 1).shift(1).fillna(0.0)

        # 3) Costs (bps) on position changes
        turnover = df["position"].diff().abs().fillna(0.0)
        costs = (self.cost_bps_single / 1e4) * turnover

        # 4) Strategy returns & equity
        df["strategy_returns"] = df["position"] * df["returns"] - costs
        df["equity_curve"] = (1 + df["strategy_returns"]).cumprod() * self.initial_capital

        # 5) Metrics
        metrics = PerformanceMetrics(df["strategy_returns"], df["equity_curve"], periods_per_year=self.ppy)
        stats = metrics.to_dict()

        return BacktestResult(
            strategy_name=strategy.name,
            ticker=price_series.ticker,
            df=df,
            stats=stats,
        )

    # ------------------------------------------------------------------
    # Pair trading
    # ------------------------------------------------------------------
    def run_pair_trading(
        self,
        strategy: PairTradingStrategy,
        ticker1: str,
        ticker2: str,
        df_prices: pd.DataFrame,
    ) -> PairBacktestResult:
        """
        Expects df_prices to contain columns:
            price_{ticker1}, price_{ticker2}

        Expects strategy.generate_pair_signals(...) to return a DataFrame indexed
        by date with at least:
            - 'position' in {-1, 0, +1}
        Optionally (for Kalman or diagnostics):
            - 'beta' (or 'hedge_ratio'), 'alpha', 'spread', 'zscore'
        """
        col1 = f"price_{ticker1}"
        col2 = f"price_{ticker2}"

        # --- 1) Clean prices & returns
        px = df_prices[[col1, col2]].astype(float).dropna()
        r1 = px[col1].pct_change().reindex(px.index).fillna(0.0)
        r2 = px[col2].pct_change().reindex(px.index).fillna(0.0)

        # --- 2) Signals (and possibly beta) from strategy
        sig = strategy.generate_pair_signals(df_prices=px.copy(), col1=col1, col2=col2).copy()

        # harmonize hedge ratio name
        if "hedge_ratio" in sig.columns:
            beta = sig["hedge_ratio"].astype(float)
        elif "beta" in sig.columns:
            beta = sig["beta"].astype(float)
        else:
            beta = pd.Series(1.0, index=sig.index, name="beta")

        pos = sig["position"].astype(float).clip(-1, 1)

        # --- 3) Align with price index & LAG one bar
        pos = pos.reindex(px.index).shift(1).fillna(0.0)
        beta = (
            beta.reindex(px.index)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(1.0)
            .clip(-self.max_leverage_beta, self.max_leverage_beta)
            .shift(1)
            .fillna(1.0)
        )

        # --- 4) Spread returns & costs
        spread_ret = (r1 - beta * r2).fillna(0.0)

        # simple costs on position changes (bps)
        turnover = pos.diff().abs().fillna(0.0)
        costs = (self.cost_bps_pair / 1e4) * turnover

        strat_ret = pos * spread_ret - costs

        # --- 5) Equity & metrics
        equity = (1 + strat_ret).cumprod() * self.initial_capital

        # Cointegration stat for info
        try:
            s1 = px[col1].loc[sig.index]
            s2 = px[col2].loc[sig.index]
            _, p_value, _ = strategy.test_cointegration(s1, s2)
        except Exception:
            p_value = np.nan

        # Build output frame (keep diagnostics if present)
        out = sig.copy()
        out = out.reindex(px.index)
        out["beta"] = beta
        out["position"] = pos
        out["spread_ret"] = spread_ret
        out["strategy_returns"] = strat_ret
        out["equity_curve"] = equity

        metrics = PerformanceMetrics(out["strategy_returns"], out["equity_curve"], periods_per_year=self.ppy)
        stats = metrics.to_dict()

        return PairBacktestResult(
            strategy_name=strategy.name,
            ticker1=ticker1,
            ticker2=ticker2,
            df=out,
            stats=stats,
            coint_pvalue=p_value,
        )
