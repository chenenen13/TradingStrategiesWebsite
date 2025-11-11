# trading/strategie_main.py
from __future__ import annotations

from abc import ABC, abstractmethod
import pandas as pd

# ---------- Base class ----------
class BaseStrategy(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def generate_signals(self, price_series) -> pd.DataFrame:
        """
        Return a DataFrame with at least a 'signal' column indicating
        buy/sell/hold signals:
          +1 = long, -1 = short, 0 = flat.
        """
        raise NotImplementedError


# Re-export concrete strategies for backward compatibility
from .momentum import MomentumStrategy  # noqa: E402
from .pair_trading import PairTradingStrategy  # noqa: E402
from .pair_kalman import KalmanPairsStrategy  # noqa: E402

__all__ = [
    "BaseStrategy",
    "MomentumStrategy",
    "PairTradingStrategy",
    "KalmanPairsStrategy",
]
