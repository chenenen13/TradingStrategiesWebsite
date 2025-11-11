# pages/strategies.py
from __future__ import annotations

from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go

from trading.momentum import MomentumStrategy
from trading.pair_trading import PairTradingStrategy
from trading.pair_kalman import KalmanPairsStrategy
from trading.backtesting import Backtester
from trading.data_loader import YahooDataProvider


# ---------- Sections (UI) ----------

def momentum_section():
    return html.Div(
        [
            html.H3("Momentum Strategy", style={"marginBottom": "12px"}),
            html.Div(
                style={
                    "display": "flex",
                    "gap": "8px",
                    "marginBottom": "16px",
                    "flexWrap": "wrap",
                },
                children=[
                    dcc.Input(
                        id="ticker",
                        type="text",
                        value="MC.PA",
                        placeholder="Ticker",
                        style={"width": "120px"},
                    ),
                    dcc.Input(
                        id="start-date",
                        type="text",
                        value="2020-01-01",
                        placeholder="Start (YYYY-MM-DD)",
                        style={"width": "150px"},
                    ),
                    dcc.Input(
                        id="end-date",
                        type="text",
                        value="2024-01-01",
                        placeholder="End (YYYY-MM-DD)",
                        style={"width": "150px"},
                    ),
                    dcc.Input(
                        id="short-window",
                        type="number",
                        value=20,
                        placeholder="Short MA",
                        style={"width": "100px"},
                    ),
                    dcc.Input(
                        id="long-window",
                        type="number",
                        value=50,
                        placeholder="Long MA",
                        style={"width": "100px"},
                    ),
                    html.Button(
                        "Run Backtest",
                        id="run-momentum",
                        style={
                            "padding": "6px 14px",
                            "borderRadius": "8px",
                            "border": "none",
                            "backgroundColor": "#2563eb",
                            "color": "white",
                            "cursor": "pointer",
                        },
                    ),
                ],
            ),
            html.Div(id="momentum-stats", style={"marginBottom": "12px"}),
            dcc.Graph(id="momentum-price-graph", style={"height": "500px"}),
            dcc.Graph(id="momentum-equity-graph", style={"height": "300px"}),
        ]
    )


def pair_trading_section():
    return html.Div(
        [
            html.H3("Pair Trading & Cointegration", style={"marginBottom": "12px"}),
            html.Div(
                style={
                    "display": "flex",
                    "gap": "8px",
                    "marginBottom": "16px",
                    "flexWrap": "wrap",
                },
                children=[
                    dcc.Input(
                        id="pair-ticker1",
                        type="text",
                        value="XLF",
                        placeholder="Ticker 1",
                        style={"width": "120px"},
                    ),
                    dcc.Input(
                        id="pair-ticker2",
                        type="text",
                        value="KBE",
                        placeholder="Ticker 2",
                        style={"width": "120px"},
                    ),
                    dcc.Input(
                        id="pair-start-date",
                        type="text",
                        value="2020-01-01",
                        placeholder="Start",
                        style={"width": "150px"},
                    ),
                    dcc.Input(
                        id="pair-end-date",
                        type="text",
                        value="2024-01-01",
                        placeholder="End",
                        style={"width": "150px"},
                    ),
                    dcc.Input(
                        id="pair-lookback",
                        type="number",
                        value=60,
                        placeholder="Lookback",
                        style={"width": "110px"},
                    ),
                    dcc.Input(
                        id="pair-z-entry",
                        type="number",
                        value=2.0,
                        placeholder="Z entry",
                        style={"width": "90px", "step": "0.1"},
                    ),
                    dcc.Input(
                        id="pair-z-exit",
                        type="number",
                        value=0.5,
                        placeholder="Z exit",
                        style={"width": "90px", "step": "0.1"},
                    ),
                    html.Button(
                        "Run Pair Backtest",
                        id="run-pair",
                        style={
                            "padding": "6px 14px",
                            "borderRadius": "8px",
                            "border": "none",
                            "backgroundColor": "#2563eb",
                            "color": "white",
                            "cursor": "pointer",
                        },
                    ),
                ],
            ),
            html.Div(id="pair-stats", style={"marginBottom": "12px"}),
            dcc.Graph(id="pair-prices-graph", style={"height": "240px"}),
            dcc.Graph(id="pair-spread-graph", style={"height": "240px"}),
            dcc.Graph(id="pair-equity-graph", style={"height": "240px"}),
        ]
    )


def pair_kalman_section():
    return html.Div(
        [
            html.H3("Pair Trading (Kalman β(t))", style={"marginBottom": "12px"}),
            html.Div(
                style={
                    "display": "flex",
                    "gap": "8px",
                    "marginBottom": "16px",
                    "flexWrap": "wrap",
                },
                children=[
                    dcc.Input(
                        id="k-ticker1",
                        type="text",
                        value="XLF",
                        placeholder="Ticker 1",
                        style={"width": "120px"},
                    ),
                    dcc.Input(
                        id="k-ticker2",
                        type="text",
                        value="KBE",
                        placeholder="Ticker 2",
                        style={"width": "120px"},
                    ),
                    dcc.Input(
                        id="k-start",
                        type="text",
                        value="2020-01-01",
                        placeholder="Start",
                        style={"width": "150px"},
                    ),
                    dcc.Input(
                        id="k-end",
                        type="text",
                        value="2024-01-01",
                        placeholder="End",
                        style={"width": "150px"},
                    ),
                    dcc.Input(
                        id="k-lookback",
                        type="number",
                        value=60,
                        placeholder="Zscore lookback",
                        style={"width": "140px"},
                    ),
                    dcc.Input(
                        id="k-z-entry",
                        type="number",
                        value=2.0,
                        placeholder="Z entry",
                        style={"width": "90px", "step": "0.1"},
                    ),
                    dcc.Input(
                        id="k-z-exit",
                        type="number",
                        value=0.5,
                        placeholder="Z exit",
                        style={"width": "90px", "step": "0.1"},
                    ),
                    dcc.Input(
                        id="k-q-alpha",
                        type="number",
                        value=1e-5,
                        placeholder="q_alpha",
                        style={"width": "110px"},
                    ),
                    dcc.Input(
                        id="k-q-beta",
                        type="number",
                        value=1e-5,
                        placeholder="q_beta",
                        style={"width": "110px"},
                    ),
                    dcc.Input(
                        id="k-r",
                        type="number",
                        value=1e-2,
                        placeholder="obs var r",
                        style={"width": "110px"},
                    ),
                    dcc.Input(
                        id="k-init",
                        type="number",
                        value=60,
                        placeholder="init window",
                        style={"width": "120px"},
                    ),
                    dcc.Input(
                        id="k-capb",
                        type="number",
                        value=5.0,
                        placeholder="cap |beta|",
                        style={"width": "120px"},
                    ),
                    html.Button(
                        "Run Kalman Pair",
                        id="run-kalman",
                        style={
                            "padding": "6px 14px",
                            "borderRadius": "8px",
                            "border": "none",
                            "backgroundColor": "#2563eb",
                            "color": "white",
                            "cursor": "pointer",
                        },
                    ),
                ],
            ),
            html.Div(id="k-stats", style={"marginBottom": "12px"}),
            dcc.Graph(id="k-prices-graph", style={"height": "220px"}),
            dcc.Graph(id="k-alpha-beta-graph", style={"height": "220px"}),
            dcc.Graph(id="k-spread-graph", style={"height": "220px"}),
            dcc.Graph(id="k-equity-graph", style={"height": "220px"}),
        ]
    )


# ---------- Page layout ----------

def layout():
    return html.Div(
        [
            html.H2("Strategies", style={"marginBottom": "20px"}),
            html.Div(
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "gap": "10px",
                    "marginBottom": "16px",
                },
                children=[
                    html.Span("Select strategy:", style={"fontWeight": 500}),
                    dcc.Dropdown(
                        id="strategy-choice",
                        options=[
                            {"label": "Momentum", "value": "momentum"},
                            {"label": "Pair Trading (Cointegration)", "value": "pair"},
                            {"label": "Pair Trading (Kalman β(t))", "value": "pair_kalman"},
                        ],
                        value="momentum",
                        clearable=False,
                        style={"width": "320px"},
                    ),
                ],
            ),
            html.Div(id="strategies-body"),
        ]
    )


# ---------- Callbacks registration ----------

def register_callbacks(app, data_provider: YahooDataProvider, backtester: Backtester):
    # Switch body by choice
    @app.callback(
        Output("strategies-body", "children"),
        Input("strategy-choice", "value"),
    )
    def render_selected_strategy(choice):
        if choice == "momentum":
            return momentum_section()
        elif choice == "pair":
            return pair_trading_section()
        elif choice == "pair_kalman":
            return pair_kalman_section()
        return html.Div("Unknown strategy")

    # -------- Momentum --------
    @app.callback(
        [
            Output("momentum-price-graph", "figure"),
            Output("momentum-equity-graph", "figure"),
            Output("momentum-stats", "children"),
        ],
        Input("run-momentum", "n_clicks"),
        [
            State("ticker", "value"),
            State("start-date", "value"),
            State("end-date", "value"),
            State("short-window", "value"),
            State("long-window", "value"),
        ],
        prevent_initial_call=True,
    )
    def run_momentum_backtest(n_clicks, ticker, start_date, end_date, short_w, long_w):
        price_series = data_provider.get_price_series(ticker, start_date, end_date)
        strategy = MomentumStrategy(short_window=int(short_w), long_window=int(long_w))
        result = backtester.run_single_asset(strategy, price_series)
        df = result.df
        stats = result.stats

        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(x=df.index, y=df["price"], name="Price"))
        fig_price.add_trace(go.Scatter(x=df.index, y=df["ma_short"], name="MA short"))
        fig_price.add_trace(go.Scatter(x=df.index, y=df["ma_long"], name="MA long"))

        signal_change = df["position"].diff().fillna(0)
        buys = df[signal_change > 0]
        sells = df[signal_change < 0]
        fig_price.add_trace(go.Scatter(
            x=buys.index, y=buys["price"], mode="markers",
            name="Buy", marker=dict(symbol="triangle-up", size=9)))
        fig_price.add_trace(go.Scatter(
            x=sells.index, y=sells["price"], mode="markers",
            name="Sell", marker=dict(symbol="triangle-down", size=9)))
        fig_price.update_layout(
            margin=dict(l=40, r=20, t=40, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        fig_equity = go.Figure()
        fig_equity.add_trace(go.Scatter(x=df.index, y=df["equity_curve"], name="Equity Curve"))
        fig_equity.update_layout(margin=dict(l=40, r=20, t=40, b=30), showlegend=False)

        stats_html = html.Ul(
            [
                html.Li(f"Total Return: {stats['total_return']:.2%}"),
                html.Li(f"Sharpe: {stats['sharpe']:.2f}"),
                html.Li(f"Max Drawdown: {stats['max_drawdown']:.2%}"),
                html.Li(f"Annualized Vol: {stats['vol_annualized']:.2%}"),
            ]
        )
        return fig_price, fig_equity, stats_html

    # -------- Pair (static beta) --------
    @app.callback(
        [
            Output("pair-prices-graph", "figure"),
            Output("pair-spread-graph", "figure"),
            Output("pair-equity-graph", "figure"),
            Output("pair-stats", "children"),
        ],
        Input("run-pair", "n_clicks"),
        [
            State("pair-ticker1", "value"),
            State("pair-ticker2", "value"),
            State("pair-start-date", "value"),
            State("pair-end-date", "value"),
            State("pair-lookback", "value"),
            State("pair-z-entry", "value"),
            State("pair-z-exit", "value"),
        ],
        prevent_initial_call=True,
    )
    def run_pair_backtest(n_clicks, t1, t2, start, end, lookback, z_entry, z_exit):
        _s1, _s2, df_prices = data_provider.get_pair(t1, t2, start, end)

        strategy = PairTradingStrategy(
            lookback=int(lookback),
            z_entry=float(z_entry),
            z_exit=float(z_exit),
        )
        result = backtester.run_pair_trading(strategy, t1, t2, df_prices)
        df = result.df
        stats = result.stats
        pval = result.coint_pvalue

        # Prices with buy/sell markers (based on position changes), markers plotted on t1 price
        fig_prices = go.Figure()
        fig_prices.add_trace(go.Scatter(x=df_prices.index, y=df_prices[f"price_{t1}"], name=t1))
        fig_prices.add_trace(go.Scatter(x=df_prices.index, y=df_prices[f"price_{t2}"], name=t2))

        px1_on_signals = df_prices[f"price_{t1}"].reindex(df.index).astype(float)
        pos = df["position"].fillna(0).astype(int)
        prev = pos.shift(1).fillna(0).astype(int)

        long_entry  = df[(prev <= 0) & (pos ==  1)]
        long_exit   = df[(prev == 1) & (pos ==  0)]
        short_entry = df[(prev >= 0) & (pos == -1)]
        short_exit  = df[(prev == -1) & (pos == 0)]

        def y_at(idx): return px1_on_signals.loc[idx]

        fig_prices.add_trace(go.Scatter(
            x=long_entry.index, y=y_at(long_entry.index),
            mode="markers", name="Long entry",
            marker=dict(symbol="triangle-up", size=10, color="#16a34a"),
        ))
        fig_prices.add_trace(go.Scatter(
            x=long_exit.index, y=y_at(long_exit.index),
            mode="markers", name="Long exit",
            marker=dict(symbol="x", size=10, color="#16a34a"),
        ))
        fig_prices.add_trace(go.Scatter(
            x=short_entry.index, y=y_at(short_entry.index),
            mode="markers", name="Short entry",
            marker=dict(symbol="triangle-down", size=10, color="#dc2626"),
        ))
        fig_prices.add_trace(go.Scatter(
            x=short_exit.index, y=y_at(short_exit.index),
            mode="markers", name="Short exit",
            marker=dict(symbol="x", size=10, color="#dc2626"),
        ))

        fig_prices.update_layout(
            margin=dict(l=40, r=20, t=40, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        fig_spread = go.Figure()
        fig_spread.add_trace(go.Scatter(x=df.index, y=df["spread"], name="Spread", yaxis="y1"))
        fig_spread.add_trace(go.Scatter(x=df.index, y=df["zscore"], name="Z-score", yaxis="y2"))
        fig_spread.update_layout(
            margin=dict(l=40, r=20, t=40, b=30),
            yaxis=dict(title="Spread"),
            yaxis2=dict(title="Z-score", overlaying="y", side="right"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        fig_equity = go.Figure()
        fig_equity.add_trace(go.Scatter(x=df.index, y=df["equity_curve"], name="Equity Curve"))
        fig_equity.update_layout(margin=dict(l=40, r=20, t=40, b=30), showlegend=False)

        stats_html = html.Ul(
            [
                html.Li(f"Cointegration p-value: {pval:.4f}"),
                html.Li(f"Total Return: {stats['total_return']:.2%}"),
                html.Li(f"Sharpe: {stats['sharpe']:.2f}"),
                html.Li(f"Max Drawdown: {stats['max_drawdown']:.2%}"),
                html.Li(f"Annualized Vol: {stats['vol_annualized']:.2%}"),
            ]
        )
        return fig_prices, fig_spread, fig_equity, stats_html

    # -------- Pair (Kalman beta(t)) --------
    @app.callback(
        [
            Output("k-prices-graph", "figure"),
            Output("k-alpha-beta-graph", "figure"),
            Output("k-spread-graph", "figure"),
            Output("k-equity-graph", "figure"),
            Output("k-stats", "children"),
        ],
        Input("run-kalman", "n_clicks"),
        [
            State("k-ticker1", "value"),
            State("k-ticker2", "value"),
            State("k-start", "value"),
            State("k-end", "value"),
            State("k-lookback", "value"),
            State("k-z-entry", "value"),
            State("k-z-exit", "value"),
            State("k-q-alpha", "value"),
            State("k-q-beta", "value"),
            State("k-r", "value"),
            State("k-init", "value"),
            State("k-capb", "value"),
        ],
        prevent_initial_call=True,
    )
    def run_pair_kalman_backtest(
        n_clicks, t1, t2, start, end, lookback, z_entry, z_exit,
        q_alpha, q_beta, r, init_w, capb
    ):
        _s1, _s2, df_prices = data_provider.get_pair(t1, t2, start, end)

        strategy = KalmanPairsStrategy(
            lookback=int(lookback),
            z_entry=float(z_entry),
            z_exit=float(z_exit),
            q_alpha=float(q_alpha),
            q_beta=float(q_beta),
            r=float(r),
            init_window=int(init_w),
            cap_beta=float(capb),
        )
        result = backtester.run_pair_trading(strategy, t1, t2, df_prices)
        df = result.df  # includes alpha/beta, spread, zscore, position, equity_curve
        stats = result.stats
        pval = result.coint_pvalue

        # Prices with buy/sell markers (on t1 price)
        fig_prices = go.Figure()
        fig_prices.add_trace(go.Scatter(x=df_prices.index, y=df_prices[f"price_{t1}"], name=t1))
        fig_prices.add_trace(go.Scatter(x=df_prices.index, y=df_prices[f"price_{t2}"], name=t2))

        px1_on_signals = df_prices[f"price_{t1}"].reindex(df.index).astype(float)
        pos = df["position"].fillna(0).astype(int)
        prev = pos.shift(1).fillna(0).astype(int)

        long_entry  = df[(prev <= 0) & (pos ==  1)]
        long_exit   = df[(prev == 1) & (pos ==  0)]
        short_entry = df[(prev >= 0) & (pos == -1)]
        short_exit  = df[(prev == -1) & (pos == 0)]

        def y_at(idx): return px1_on_signals.loc[idx]

        fig_prices.add_trace(go.Scatter(
            x=long_entry.index, y=y_at(long_entry.index),
            mode="markers", name="Long entry",
            marker=dict(symbol="triangle-up", size=10, color="#16a34a"),
        ))
        fig_prices.add_trace(go.Scatter(
            x=long_exit.index, y=y_at(long_exit.index),
            mode="markers", name="Long exit",
            marker=dict(symbol="x", size=10, color="#16a34a"),
        ))
        fig_prices.add_trace(go.Scatter(
            x=short_entry.index, y=y_at(short_entry.index),
            mode="markers", name="Short entry",
            marker=dict(symbol="triangle-down", size=10, color="#dc2626"),
        ))
        fig_prices.add_trace(go.Scatter(
            x=short_exit.index, y=y_at(short_exit.index),
            mode="markers", name="Short exit",
            marker=dict(symbol="x", size=10, color="#dc2626"),
        ))

        fig_prices.update_layout(
            margin=dict(l=40, r=20, t=40, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        # Alpha/Beta graph
        fig_ab = go.Figure()
        if "alpha" in df.columns:
            fig_ab.add_trace(go.Scatter(x=df.index, y=df["alpha"], name="alpha(t)"))
        if "beta" in df.columns:
            fig_ab.add_trace(go.Scatter(x=df.index, y=df["beta"], name="beta(t)"))
        fig_ab.update_layout(
            margin=dict(l=40, r=20, t=40, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        # Spread + Z graph
        fig_spread = go.Figure()
        if "spread" in df.columns:
            fig_spread.add_trace(go.Scatter(x=df.index, y=df["spread"], name="Spread", yaxis="y1"))
        if "zscore" in df.columns:
            fig_spread.add_trace(go.Scatter(x=df.index, y=df["zscore"], name="Z-score", yaxis="y2"))
        fig_spread.update_layout(
            margin=dict(l=40, r=20, t=40, b=30),
            yaxis=dict(title="Spread"),
            yaxis2=dict(title="Z-score", overlaying="y", side="right"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        # Equity curve
        fig_equity = go.Figure()
        fig_equity.add_trace(go.Scatter(x=df.index, y=df["equity_curve"], name="Equity Curve"))
        fig_equity.update_layout(margin=dict(l=40, r=20, t=40, b=30), showlegend=False)

        stats_html = html.Ul(
            [
                html.Li(f"Cointegration p-value: {pval:.4f}"),
                html.Li(f"Total Return: {stats['total_return']:.2%}"),
                html.Li(f"Sharpe: {stats['sharpe']:.2f}"),
                html.Li(f"Max Drawdown: {stats['max_drawdown']:.2%}"),
                html.Li(f"Annualized Vol: {stats['vol_annualized']:.2%}"),
            ]
        )
        return fig_prices, fig_ab, fig_spread, fig_equity, stats_html
