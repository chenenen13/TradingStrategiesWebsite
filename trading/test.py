# trading/test.py
import pandas as pd
import yfinance as yf

TICKER = "AAPL"
LIMIT = 16

def _to_naive_utc(ts: pd.Timestamp) -> pd.Timestamp:
    """Return a tz-naive UTC timestamp (safe for comparisons)."""
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None and ts.tz is not None:
        # convert to UTC then drop tz info
        return ts.tz_convert("UTC").tz_localize(None)
    return ts  # already tz-naive

def get_eps_surprises(ticker: str, limit: int = 16) -> pd.DataFrame:
    t = yf.Ticker(ticker)
    df = t.get_earnings_dates(limit=limit)

    if df is None or len(df) == 0:
        raise RuntimeError("yfinance returned no earnings_dates")

    # --- Ensure a 'date' column exists ---
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()

    date_col = None
    for cand in ("Earnings Date", "Date", "index"):
        if cand in df.columns:
            date_col = cand
            break
    if date_col is None:
        # last resort: any datetime-like column
        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                date_col = c
                break
    if date_col is None:
        raise KeyError(f"Can't find earnings date column in {list(df.columns)}")

    df = df.rename(columns={date_col: "date"})

    # --- Normalize numeric column names ---
    col_map = {}
    for c in df.columns:
        lc = c.lower()
        if "estimate" in lc and "eps" in lc:
            col_map[c] = "EPS Estimate"
        elif "reported" in lc and "eps" in lc:
            col_map[c] = "Reported EPS"
        elif "surprise" in lc:
            col_map[c] = "Surprise(%)"
    df = df.rename(columns=col_map)

    keep = ["date", "EPS Estimate", "Reported EPS", "Surprise(%)"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()

    # dates -> pandas datetime then tz-naive UTC
    df["date"] = pd.to_datetime(df["date"], errors="coerce").map(_to_naive_utc)
    df = df.dropna(subset=["date"])

    # Compute surprise if not provided
    if "Surprise(%)" not in df.columns and {"EPS Estimate", "Reported EPS"}.issubset(df.columns):
        est = pd.to_numeric(df["EPS Estimate"], errors="coerce")
        rep = pd.to_numeric(df["Reported EPS"], errors="coerce")
        df["Surprise(%)"] = (rep - est).div(est.abs()).mul(100)

    df = df.sort_values("date", ascending=False).reset_index(drop=True)
    return df

def next_open_gap_pct(ticker: str, earn_ts: pd.Timestamp):
    """
    Gap d'ouverture (%) entre la clôture précédente (<= date earnings)
    et l'ouverture de la séance suivante (> date earnings).
    Toutes les dates sont comparées en tz-naïf.
    """
    d0 = _to_naive_utc(earn_ts).floor("D")
    # fenêtre large pour couvrir WE/jours fériés
    start = (d0 - pd.Timedelta(days=7)).date().isoformat()
    end   = (d0 + pd.Timedelta(days=10)).date().isoformat()

    try:
        px = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False, interval="1d")
    except Exception:
        return pd.NA
    if px.empty:
        return pd.NA

    # sécurise l'index en tz-naïf (au cas où)
    if getattr(px.index, "tz", None) is not None:
        px.index = px.index.tz_convert("UTC").tz_localize(None)

    # repère la dernière clôture <= earnings, et la 1ère ouverture > earnings
    try:
        prev = px.index[px.index <= d0].max()
        nxt  = px.index[px.index >  d0].min()
    except ValueError:
        return pd.NA

    if pd.isna(prev) or pd.isna(nxt):
        return pd.NA

    # gap = Open(next session) / Close(prev session) - 1
    if "Open" not in px.columns or "Close" not in px.columns:
        return pd.NA
    try:
        return (px.loc[nxt, "Open"] / px.loc[prev, "Close"] - 1) * 100
    except Exception:
        return pd.NA

if __name__ == "__main__":
    df = get_eps_surprises(TICKER, LIMIT)
    # Ajoute le gap d'ouverture (peut être long → commente si besoin)
    df["Gap next open (%)"] = df["date"].apply(lambda d: next_open_gap_pct(TICKER, d))
    print(df)
