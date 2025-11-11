# trading/models.py
from dataclasses import dataclass
import pandas as pd
from typing import Dict, Any

@dataclass
class PriceSeries:
    ticker: str
    data: pd.DataFrame  # colonnes: ['price'] au minimum

@dataclass
class BacktestResult:
    strategy_name: str
    ticker: str
    df: pd.DataFrame  # DataFrame final avec returns, equity_curve, signaux, etc.
    stats: Dict[str, Any]  # Sharpe, max DD, etc.

@dataclass
class PairBacktestResult:
    strategy_name: str
    ticker1: str
    ticker2: str
    df: pd.DataFrame
    stats: Dict[str, Any]
    coint_pvalue: float
