# trading/data_loader.py
from __future__ import annotations

from typing import List, Optional
import os
import pandas as pd
import yfinance as yf

# robustesse réseau
import requests
import requests_cache
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .models import PriceSeries


# ----------------------------- Helpers -----------------------------
def _make_cached_session() -> requests.Session:
    """
    Crée une session HTTP avec cache sur /tmp (autorisé sur Render).
    Cela réduit les rate-limits Yahoo et accélère les cold starts.
    """
    cache_dir = "/tmp/yf_cache"
    os.makedirs(cache_dir, exist_ok=True)
    return requests_cache.CachedSession(
        cache_name=os.path.join(cache_dir, "http_cache"),
        backend="sqlite",
        expire_after=3600,  # 1h de cache
    )


def _tz_naive_index(df: pd.DataFrame) -> pd.DataFrame:
    """Rend l'index tz-naive pour éviter les merges vides."""
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        # tz_convert(None) si tz aware; sinon tz_localize(None) plante.
        df.index = df.index.tz_convert(None)
    return df


# ======================== Data Providers ===========================
class MarketDataProvider:
    """Abstract base class for market data providers."""
    def get_price_series(self, ticker: str, start: str, end: str) -> PriceSeries:
        raise NotImplementedError

    def get_multiple_price_series(self, tickers: List[str], start: str, end: str) -> dict:
        return {t: self.get_price_series(t, start, end) for t in tickers}


class YahooDataProvider(MarketDataProvider):
    """
    Provider Yahoo Finance robuste pour PaaS (Render):
      - session HTTP mise en cache (/tmp)
      - retries exponentiels
      - downloads avec threads=False
      - nettoyage tz-naive
    """
    def __init__(self) -> None:
        self.session = _make_cached_session()

    # --------------- low-level wrapper avec retries ----------------
    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=1, max=8),
        retry=retry_if_exception_type(Exception),
    )
    def _download(self, ticker: str, **kwargs) -> pd.DataFrame:
        """
        Enveloppe yf.download avec:
         - session cache
         - threads=False
         - auto_adjust défini par le caller (kwargs)
        Léve RuntimeError si DataFrame vide.
        """
        kwargs = dict(kwargs)
        kwargs.setdefault("progress", False)
        kwargs.setdefault("threads", False)
        kwargs.setdefault("session", self.session)

        df = yf.download(ticker, **kwargs)
        if df is None or df.empty:
            raise RuntimeError(f"Empty data from Yahoo for {ticker} with {kwargs}")
        return _tz_naive_index(df)

    # --------------------------- Prices ---------------------------
    def get_price_series(self, ticker: str, start: str, end: str) -> PriceSeries:
        # On garde auto_adjust=False pour la cohérence avec le reste du projet
        raw = self._download(
            ticker,
            start=start,
            end=end,
            auto_adjust=False,
            interval="1d",
        )
        # colonne de référence = 'Close' → 'price'
        if "Close" not in raw.columns:
            # fallback: prend la 1re colonne numérique
            num_cols = [c for c in raw.columns if pd.api.types.is_numeric_dtype(raw[c])]
            if not num_cols:
                raise RuntimeError(f"No price-like column for {ticker}")
            close = raw[num_cols[0]].rename("price")
        else:
            close = raw["Close"].rename("price")

        data = pd.DataFrame(close).dropna()
        if data.empty:
            raise RuntimeError(f"No usable price data for {ticker}")

        return PriceSeries(ticker=ticker, data=data)

    # --------------------------- Pair -----------------------------
    def get_pair(self, ticker1: str, ticker2: str, start: str, end: str):
        """
        Return (PriceSeries for ticker1, PriceSeries for ticker2, joined DataFrame).
        The joined DataFrame has columns: price_<ticker1>, price_<ticker2>.
        """
        s1 = self.get_price_series(ticker1, start, end)
        s2 = self.get_price_series(ticker2, start, end)

        df = s1.data.join(
            s2.data,
            how="inner",
            lsuffix=f"_{ticker1}",
            rsuffix=f"_{ticker2}",
        ).dropna()
        return s1, s2, df

    # ------------------------- Dividends --------------------------
    def get_dividends(self, ticker: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.Series:
        t = yf.Ticker(ticker, session=self.session)
        divs = t.dividends  # Series

        if divs is None or divs.empty:
            return pd.Series(dtype="float64")

        if isinstance(divs.index, pd.DatetimeIndex) and divs.index.tz is not None:
            divs.index = divs.index.tz_convert(None)

        if start is not None:
            divs = divs[divs.index >= pd.to_datetime(start)]
        if end is not None:
            divs = divs[divs.index <= pd.to_datetime(end)]
        return divs

    def get_next_dividend_info(self, ticker: str) -> dict:
        t = yf.Ticker(ticker, session=self.session)
        info = getattr(t, "info", {}) or {}

        ex_date = None
        ex_ts = info.get("exDividendDate")
        if ex_ts:
            try:
                ex_date = pd.to_datetime(ex_ts, unit="s").date()
            except Exception:
                ex_date = None

        annual_rate = info.get("dividendRate")
        dividend_yield = info.get("dividendYield")
        forecast_amount = (annual_rate / 4.0) if isinstance(annual_rate, (int, float)) else None

        return {
            "ex_date": ex_date,
            "annual_rate": annual_rate,
            "yield": dividend_yield,
            "forecast_amount": forecast_amount,
        }

    # -------------------------- Earnings --------------------------
    def get_earnings(self, ticker: str) -> pd.DataFrame:
        """
        Construit un tableau earnings à partir des income statements:
          - Period, Revenue, Earnings, Type ("Yearly"/"Quarterly")
        """
        t = yf.Ticker(ticker, session=self.session)
        frames: list[pd.DataFrame] = []

        # yearly
        yearly = getattr(t, "income_stmt", None)
        if isinstance(yearly, pd.DataFrame) and not yearly.empty:
            df_y = yearly.T.copy()
            ren = {}
            if "Total Revenue" in df_y.columns: ren["Total Revenue"] = "Revenue"
            if "Net Income" in df_y.columns: ren["Net Income"] = "Earnings"
            df_y = df_y.rename(columns=ren)
            keep = [c for c in ["Revenue", "Earnings"] if c in df_y.columns]
            if keep:
                df_y = df_y[keep].copy()
                idx = df_y.index
                period = idx.year.astype(str) if isinstance(idx, pd.DatetimeIndex) else idx.astype(str)
                df_y["Period"] = period
                df_y["Type"] = "Yearly"
                frames.append(df_y)

        # quarterly
        q_stmt = getattr(t, "quarterly_income_stmt", None)
        if isinstance(q_stmt, pd.DataFrame) and not q_stmt.empty:
            df_q = q_stmt.T.copy()
            ren = {}
            if "Total Revenue" in df_q.columns: ren["Total Revenue"] = "Revenue"
            if "Net Income" in df_q.columns: ren["Net Income"] = "Earnings"
            df_q = df_q.rename(columns=ren)
            keep = [c for c in ["Revenue", "Earnings"] if c in df_q.columns]
            if keep:
                df_q = df_q[keep].copy()
                idx = df_q.index
                period = idx.to_period("Q").astype(str) if isinstance(idx, pd.DatetimeIndex) else idx.astype(str)
                df_q["Period"] = period
                df_q["Type"] = "Quarterly"
                frames.append(df_q)

        if not frames:
            return pd.DataFrame(columns=["Period", "Revenue", "Earnings", "Type"])

        out = pd.concat(frames, axis=0, ignore_index=True)
        cols = [c for c in ["Period", "Revenue", "Earnings", "Type"] if c in out.columns]
        return out[cols]

    # -------------------- EPS surprises (base) --------------------
    def get_eps_surprises(self, ticker: str, limit: Optional[int] = 10) -> pd.DataFrame:
        t = yf.Ticker(ticker, session=self.session)

        try:
            hist = t.get_earnings_dates(limit=limit)
        except Exception:
            hist = None

        if hist is None or len(hist) == 0:
            return pd.DataFrame()

        df = hist.reset_index(drop=False) if not isinstance(hist, pd.DataFrame) else hist.copy()
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()

        # détecter colonne de date
        date_col = None
        for c in ("Earnings Date", "Date", "index"):
            if c in df.columns:
                date_col = c
                break
        if date_col is None:
            for c in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[c]):
                    date_col = c
                    break
        if date_col is None:
            return pd.DataFrame()

        # renommer colonnes utiles
        ren = {}
        for c in df.columns:
            lc = str(c).lower()
            if "estimate" in lc and "eps" in lc:
                ren[c] = "EPS Estimate"
            elif ("reported" in lc and "eps" in lc) or ("actual" in lc and "eps" in lc):
                ren[c] = "Reported EPS"
            elif "surprise" in lc:
                ren[c] = "Surprise(%)"
            elif "quarter" in lc or "period" in lc:
                ren[c] = "Period"
        df = df.rename(columns=ren)

        out = pd.DataFrame()
        out["date"] = pd.to_datetime(df[date_col], errors="coerce")
        out["period"] = df["Period"].astype(str) if "Period" in df.columns else out["date"].dt.strftime("%Y-%m-%d")
        out["eps_estimate"] = pd.to_numeric(df.get("EPS Estimate"), errors="coerce")
        out["eps_actual"] = pd.to_numeric(df.get("Reported EPS"), errors="coerce")

        if "Surprise(%)" in df.columns:
            out["surprise_pct"] = pd.to_numeric(df["Surprise(%)"], errors="coerce")
        else:
            out["surprise_pct"] = (out["eps_actual"] - out["eps_estimate"]).div(out["eps_estimate"].abs()).mul(100)

        out = out.dropna(subset=["date"]).sort_values("date", ascending=False)
        if limit:
            out = out.head(limit)
        return out.reset_index(drop=True)

    # -------- EPS surprises + next-session opening gap ------------
    def get_eps_surprises_with_gaps(
        self,
        ticker: str,
        limit: int = 16,
        include_gap: bool = True,
    ) -> pd.DataFrame:
        import numpy as np

        base = self.get_eps_surprises(ticker, limit=limit if limit else None)
        if base is None or base.empty:
            return base

        df = base.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        earn_day = df["date"].dt.normalize()

        if not include_gap:
            return df.reset_index(drop=True)

        # fenêtre prix autour des dates d'earnings
        w_start = (earn_day.min() - pd.Timedelta(days=7)).date().isoformat()
        w_end = (earn_day.max() + pd.Timedelta(days=10)).date().isoformat()

        # close via pipeline standard
        ps_close = self.get_price_series(ticker, w_start, w_end).data.copy()
        if ps_close.empty:
            df["gap_next_open_pct"] = pd.NA
            return df.reset_index(drop=True)

        # OHLC pour vrais opens
        ohlc = self._download(
            ticker,
            start=w_start,
            end=w_end,
            auto_adjust=False,
            interval="1d",
        )
        if ohlc.empty or not {"Open", "Close"}.issubset(ohlc.columns):
            df["gap_next_open_pct"] = pd.NA
            return df.reset_index(drop=True)

        px = ohlc[["Open", "Close"]].dropna().copy()
        px["prev_close"] = px["Close"].shift(1)
        px = px.dropna(subset=["prev_close"])
        px["day"] = px.index.normalize()
        px["gap_pct"] = (px["Open"] / px["prev_close"] - 1.0) * 100.0

        day_to_gap = px.set_index("day")["gap_pct"]
        days = day_to_gap.index.values  # DatetimeIndex trié

        def gap_for_calendar_day(d):
            if d in day_to_gap.index:
                return float(day_to_gap.loc[d])
            i = np.searchsorted(days, np.datetime64(d), side="right")
            if i >= len(days):
                return pd.NA
            return float(day_to_gap.iloc[i])

        df["gap_next_open_pct"] = earn_day.map(gap_for_calendar_day)
        return df.reset_index(drop=True)

    # ------------------------ Opening Gaps ------------------------
    def get_opening_gaps(
        self,
        ticker: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: Optional[int] = 20,
        only_earnings: bool = False,
    ) -> pd.DataFrame:
        px = self._download(
            ticker,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=False,
        )

        if px.empty or not {"Open", "Close"}.issubset(px.columns):
            return pd.DataFrame(columns=[
                "date", "prev_close", "open", "close", "gap_pct", "session_return_pct"
            ])

        px = px[["Open", "Close"]].dropna().copy()
        px["prev_close"] = px["Close"].shift(1)
        px = px.dropna(subset=["prev_close"])

        px["gap_pct"] = (px["Open"] / px["prev_close"] - 1.0) * 100.0
        px["session_return_pct"] = (px["Close"] / px["Open"] - 1.0) * 100.0

        out = px[["Open", "Close", "prev_close", "gap_pct", "session_return_pct"]].copy()
        out = out.rename(columns={"Open": "open", "Close": "close"})
        out.index.name = "date"
        out = out.reset_index()

        if only_earnings:
            eps = self.get_eps_surprises(ticker, limit=None)
            if not eps.empty:
                earnings_days = pd.to_datetime(eps["date"]).dt.normalize().unique()
                out = out[out["date"].isin(earnings_days)]

        if limit:
            out = out.reindex(out["gap_pct"].abs().sort_values(ascending=False).index).head(limit)

        return out[["date", "prev_close", "open", "close", "gap_pct", "session_return_pct"]]

    # --------------------------- News -----------------------------
    def get_news(self, ticker: str, limit: int = 10) -> list[dict]:
        t = yf.Ticker(ticker, session=self.session)
        news = getattr(t, "news", []) or []
        return news[:limit]

    # ---------------------- Global Indices ------------------------
    def get_global_indices_snapshot(self):
        """
        Snapshot d’indices majeurs (dernier prix & % jour).
        """
        index_map = {
            "S&P 500": "^GSPC",
            "CAC 40": "^FCHI",
            "Euro Stoxx 50": "^STOXX50E",
            "Hang Seng": "^HSI",
            "Shanghai Composite": "000001.SS",
            "Nikkei 225": "^N225",
        }

        snapshot = []
        for name, ticker in index_map.items():
            try:
                hist = self._download(
                    ticker,
                    period="2d",
                    interval="1d",
                    auto_adjust=False,
                )
            except Exception:
                continue

            if hist.empty or "Close" not in hist.columns:
                continue

            close = hist["Close"]
            last = float(close.iloc[-1])
            prev = float(close.iloc[-2]) if len(close) > 1 else last
            pct = (last / prev - 1.0) * 100.0 if prev != 0.0 else 0.0

            snapshot.append(
                {"name": name, "ticker": ticker, "last": last, "change_pct": pct}
            )

        return snapshot
