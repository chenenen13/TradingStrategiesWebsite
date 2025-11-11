# trading/metrics.py
import pandas as pd
import numpy as np

class PerformanceMetrics:
    def __init__(self, strategy_returns: pd.Series, equity_curve: pd.Series, periods_per_year: int = 252):
        self.returns = strategy_returns
        self.equity_curve = equity_curve
        self.periods_per_year = periods_per_year

    def sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        excess = self.returns - risk_free_rate / self.periods_per_year
        if excess.std() == 0:
            return 0.0
        return np.sqrt(self.periods_per_year) * excess.mean() / excess.std()

    def max_drawdown(self) -> float:
        running_max = self.equity_curve.cummax()
        drawdown = self.equity_curve / running_max - 1.0
        return float(drawdown.min())

    def total_return(self) -> float:
        return float(self.equity_curve.iloc[-1] / self.equity_curve.iloc[0] - 1.0)

    def annualized_vol(self) -> float:
        return float(np.sqrt(self.periods_per_year) * self.returns.std())

    def to_dict(self):
        return {
            "total_return": self.total_return(),
            "sharpe": self.sharpe_ratio(),
            "max_drawdown": self.max_drawdown(),
            "vol_annualized": self.annualized_vol(),
        }
