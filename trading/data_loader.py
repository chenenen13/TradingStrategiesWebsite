# trading/data_loader.py
from typing import List, Optional
import pandas as pd
import yfinance as yf

from .models import PriceSeries


class MarketDataProvider:
    """Abstract base class for market data providers."""
    def get_price_series(self, ticker: str, start: str, end: str) -> PriceSeries:
        raise NotImplementedError

    def get_multiple_price_series(self, tickers: List[str], start: str, end: str) -> dict:
        return {t: self.get_price_series(t, start, end) for t in tickers}


class YahooDataProvider(MarketDataProvider):
    # --------------------------- Prices ---------------------------
    def get_price_series(self, ticker: str, start: str, end: str) -> PriceSeries:
        data = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
        )

        # yfinance peut renvoyer un DataFrame vide sans lever d'exception
        if data is None or data.empty:
            print(f"[WARNING] No data returned for {ticker} between {start} and {end}", file=sys.stderr, flush=True)
            # on retourne une PriceSeries vide
            empty = pd.DataFrame(columns=["price"])
            return PriceSeries(ticker=ticker, data=empty)

        if "Close" not in data.columns:
            print(f"[WARNING] 'Close' column missing for {ticker}", file=sys.stderr, flush=True)
            empty = pd.DataFrame(columns=["price"])
            return PriceSeries(ticker=ticker, data=empty)

        data = data[["Close"]].dropna()
        data.columns = ["price"]
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
        )
        return s1, s2, df

    # ------------------------- Dividends --------------------------
    def get_dividends(self, ticker: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.Series:
        t = yf.Ticker(ticker)
        divs = t.dividends  # Series indexed by date

        if divs is None or divs.empty:
            return divs

        # Make the index timezone-naive to avoid tz-aware vs tz-naive comparison issues
        idx = divs.index
        if hasattr(idx, "tz") and idx.tz is not None:
            divs.index = idx.tz_convert(None)

        if start is not None:
            start_ts = pd.to_datetime(start)
            divs = divs[divs.index >= start_ts]

        if end is not None:
            end_ts = pd.to_datetime(end)
            divs = divs[divs.index <= end_ts]

        return divs

    def get_next_dividend_info(self, ticker: str) -> dict:
        """
        Return info about next dividend if available via yfinance .info:
        - ex_date: date of next ex-dividend (or None)
        - annual_rate: indicated annual dividend per share (or None)
        - yield: indicated dividend yield (decimal, not %)
        - forecast_amount: naive estimate of next dividend (annual_rate / 4)
        """
        t = yf.Ticker(ticker)
        info = getattr(t, "info", {}) or {}

        # exDividendDate is usually a unix timestamp (seconds)
        ex_date = None
        ex_ts = info.get("exDividendDate")
        if ex_ts:
            try:
                ex_date = pd.to_datetime(ex_ts, unit="s").date()
            except Exception:
                ex_date = None

        annual_rate = info.get("dividendRate")
        dividend_yield = info.get("dividendYield")

        forecast_amount = None
        if isinstance(annual_rate, (int, float)):
            # Very naive assumption: quarterly payments
            forecast_amount = annual_rate / 4.0

        return {
            "ex_date": ex_date,
            "annual_rate": annual_rate,
            "yield": dividend_yield,
            "forecast_amount": forecast_amount,
        }

    # -------------------------- Earnings --------------------------
    def get_earnings(self, ticker: str) -> pd.DataFrame:
        """
        Build an 'earnings-like' table from income statements:

        - Yearly: Ticker.income_stmt        (rows = metrics, cols = dates)
        - Quarterly: Ticker.quarterly_income_stmt

        Keep:
        - Period  (year or quarter string)
        - Revenue (Total Revenue)
        - Earnings (Net Income)
        - Type ("Yearly" / "Quarterly")
        """
        t = yf.Ticker(ticker)
        frames = []

        # ----- Yearly -----
        yearly = getattr(t, "income_stmt", None)
        if isinstance(yearly, pd.DataFrame) and not yearly.empty:
            df_y = yearly.T  # rows = dates, cols = metrics
            rename_map = {}
            if "Total Revenue" in df_y.columns:
                rename_map["Total Revenue"] = "Revenue"
            if "Net Income" in df_y.columns:
                rename_map["Net Income"] = "Earnings"
            df_y = df_y.rename(columns=rename_map)

            keep_cols = [c for c in ["Revenue", "Earnings"] if c in df_y.columns]
            if keep_cols:
                df_y = df_y[keep_cols].copy()
                idx = df_y.index
                period = idx.year.astype(str) if isinstance(idx, pd.DatetimeIndex) else idx.astype(str)
                df_y["Period"] = period
                df_y["Type"] = "Yearly"
                frames.append(df_y)

        # ----- Quarterly -----
        q_stmt = getattr(t, "quarterly_income_stmt", None)
        if isinstance(q_stmt, pd.DataFrame) and not q_stmt.empty:
            df_q = q_stmt.T
            rename_map = {}
            if "Total Revenue" in df_q.columns:
                rename_map["Total Revenue"] = "Revenue"
            if "Net Income" in df_q.columns:
                rename_map["Net Income"] = "Earnings"
            df_q = df_q.rename(columns=rename_map)

            keep_cols = [c for c in ["Revenue", "Earnings"] if c in df_q.columns]
            if keep_cols:
                df_q = df_q[keep_cols].copy()
                idx = df_q.index
                period = idx.to_period("Q").astype(str) if isinstance(idx, pd.DatetimeIndex) else idx.astype(str)
                df_q["Period"] = period
                df_q["Type"] = "Quarterly"
                frames.append(df_q)

        if not frames:
            return pd.DataFrame(columns=["Period", "Revenue", "Earnings", "Type"])

        earnings = pd.concat(frames, axis=0, ignore_index=True)
        cols_order = [c for c in ["Period", "Revenue", "Earnings", "Type"] if c in earnings.columns]
        return earnings[cols_order]

    # -------------------- EPS surprises (base) --------------------
    def get_eps_surprises(self, ticker: str, limit: Optional[int] = 10) -> pd.DataFrame:
        """
        Fetch EPS estimate vs actual surprises for a ticker.

        Returns columns:
        - date (datetime)
        - period (e.g., '2024Q4')
        - eps_estimate
        - eps_actual
        - surprise_pct
        """
        t = yf.Ticker(ticker)

        try:
            hist = t.get_earnings_dates(limit=limit)
        except Exception:
            hist = None

        if hist is None or len(hist) == 0:
            return pd.DataFrame()

        df = hist.reset_index(drop=False) if not isinstance(hist, pd.DataFrame) else hist.copy()
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()

        # Try to detect columns
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

        # Normalize columns
        rename_map = {}
        for c in df.columns:
            lc = str(c).lower()
            if "estimate" in lc and "eps" in lc:
                rename_map[c] = "EPS Estimate"
            elif ("reported" in lc and "eps" in lc) or ("actual" in lc and "eps" in lc):
                rename_map[c] = "Reported EPS"
            elif "surprise" in lc:
                rename_map[c] = "Surprise(%)"
            elif "quarter" in lc or "period" in lc:
                rename_map[c] = "Period"
        df = df.rename(columns=rename_map)

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
        """
        EPS surprises + next-session open gap (%) computed from the same
        price source used in Strategies (via get_price_series).

        Gap rule:
          gap_next_open_pct = (Open[next_trading_day] / Close[prev_trading_day] - 1) * 100
        where:
          prev_trading_day = last trading day <= earnings calendar day
          next_trading_day = first trading day  > earnings calendar day
        """
        import numpy as np

        # 1) Base EPS surprises (dates + estimates/actuals)
        base = self.get_eps_surprises(ticker, limit=limit if limit else None)
        if base is None or base.empty:
            return base

        df = base.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        earn_day = df["date"].dt.normalize()

        if not include_gap:
            return df.reset_index(drop=True)

        # 2) One price download using the same pipeline as Strategies
        #    We grab a safe window around all earnings dates.
        w_start = (earn_day.min() - pd.Timedelta(days=7)).date().isoformat()
        w_end   = (earn_day.max() + pd.Timedelta(days=10)).date().isoformat()

        # close series (from get_price_series = yf.download(..., auto_adjust=False) then Close→price)
        ps_close = self.get_price_series(ticker, w_start, w_end).data.copy()  # columns: ['price'] (Close)
        if ps_close.empty:
            df["gap_next_open_pct"] = pd.NA
            return df.reset_index(drop=True)

        # also fetch raw daily OHLC once to get real Open (same date window)
        ohlc = yf.download(
            ticker, start=w_start, end=w_end,
            interval="1d", auto_adjust=False, progress=False
        )
        if ohlc.empty or not {"Open","Close"}.issubset(ohlc.columns):
            df["gap_next_open_pct"] = pd.NA
            return df.reset_index(drop=True)

        # normalize indices (tz-naive) & keep needed cols
        if getattr(ohlc.index, "tz", None) is not None:
            ohlc.index = ohlc.index.tz_convert(None)
        if getattr(ps_close.index, "tz", None) is not None:
            ps_close.index = ps_close.index.tz_convert(None)

        px = ohlc[["Open","Close"]].dropna().copy()
        px["prev_close"] = px["Close"].shift(1)
        px = px.dropna(subset=["prev_close"])
        px["day"] = px.index.normalize()

        # compute gap for each trading day (using that day's Open and previous day's Close)
        px["gap_pct"] = (px["Open"] / px["prev_close"] - 1.0) * 100.0
        day_to_gap = px.set_index("day")["gap_pct"]
        days = day_to_gap.index.values  # sorted DatetimeIndex

        def gap_for_calendar_day(d):
            # if earnings day is trading day, use that day's gap; else first day strictly after
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
        """
        Compute historical market-open gaps.

        Returns:
          date, prev_close, open, close, gap_pct, session_return_pct
        """
        px = yf.download(
            ticker,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=False,
            progress=False,
        )

        if px.empty or not {"Open", "Close"}.issubset(px.columns):
            return pd.DataFrame(columns=[
                "date", "prev_close", "open", "close", "gap_pct", "session_return_pct"
            ])

        if hasattr(px.index, "tz") and px.index.tz is not None:
            px.index = px.index.tz_convert(None)

        px = px[["Open", "Close"]].dropna().copy()
        px["prev_close"] = px["Close"].shift(1)
        px = px.dropna(subset=["prev_close"])

        px["gap_pct"] = (px["Open"] / px["prev_close"] - 1.0) * 100.0
        px["session_return_pct"] = (px["Close"] / px["Open"] - 1.0) * 100.0

        out = px[["prev_close", "Open", "Close", "gap_pct", "session_return_pct"]].copy()
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
        t = yf.Ticker(ticker)
        news = getattr(t, "news", []) or []
        return news[:limit]

    # ---------------------- Global Indices ------------------------
    def get_global_indices_snapshot(self):
        """
        Return snapshot for major indices (last price & daily % change).
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
            hist = yf.download(
                ticker,
                period="2d",
                interval="1d",
                auto_adjust=False,
                progress=False,
            )
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
