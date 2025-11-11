from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
from trading.data_loader import YahooDataProvider
from plotly.subplots import make_subplots

def layout():
    return html.Div(
        [
            html.H2("Dividends", style={"marginBottom": "16px"}),
            html.Div(
                [
                    dcc.Input(
                        id="div-ticker",
                        type="text",
                        value="AAPL",
                        placeholder="Ticker",
                        style={"width": "120px", "marginRight": "8px"},
                    ),
                    dcc.Input(
                        id="div-start",
                        type="text",
                        value="2018-01-01",
                        placeholder="Start",
                        style={"width": "130px", "marginRight": "8px"},
                    ),
                    dcc.Input(
                        id="div-end",
                        type="text",
                        value="2024-01-01",
                        placeholder="End",
                        style={"width": "130px", "marginRight": "8px"},
                    ),
                    html.Button(
                        "Load Dividends",
                        id="div-load",
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
                style={"marginBottom": "16px"},
            ),

            # bloc info
            html.Div(id="div-info", style={"marginBottom": "16px"}),

            dcc.Graph(id="div-graph", style={"height": "620px"}),
        ]
    )


def register_callbacks(app, data_provider: YahooDataProvider):
    @app.callback(
        [Output("div-graph", "figure"), Output("div-info", "children")],
        Input("div-load", "n_clicks"),
        [
            State("div-ticker", "value"),
            State("div-start", "value"),
            State("div-end", "value"),
        ],
        prevent_initial_call=True,
    )
    def load_dividends(n_clicks, ticker, start, end):
        # 1) data
        divs = data_provider.get_dividends(ticker, start, end)
        price_series = data_provider.get_price_series(ticker, start, end)
        price_df = price_series.data

        # --- figure: 2 rows, shared x ---
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.65, 0.35],
            subplot_titles=("Price", "Dividends"),
        )

        # price (top)
        if not price_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=price_df.index,
                    y=price_df["price"],
                    name="Price",
                    mode="lines",
                    line=dict(color="red"),
                ),
                row=1,
                col=1,
            )

        # dividends (bottom)
        if divs is not None and not divs.empty:
            bar_width_ms = 10 * 24 * 60 * 60 * 1000  # 10 days
            fig.add_trace(
                go.Bar(
                    x=divs.index,
                    y=divs.values,
                    name="Dividend",
                    marker=dict(color="rgba(59,130,246,0.85)"),
                    width=[bar_width_ms] * len(divs),
                ),
                row=2,
                col=1,
            )

        if (divs is None or divs.empty) and price_df.empty:
            fig.update_layout(
                title="No data for this period",
                margin=dict(l=40, r=20, t=40, b=30),
            )
        else:
            fig.update_layout(
                margin=dict(l=40, r=20, t=40, b=30),
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                ),
            )
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Price", row=1, col=1)
            fig.update_yaxes(title_text="Dividend per share", row=2, col=1)

        # 2) info section stays exactly like avant
        info = data_provider.get_next_dividend_info(ticker)
        items = []

        if info.get("ex_date"):
            items.append(html.Li(f"Next ex-dividend date: {info['ex_date']}"))

        if info.get("forecast_amount") is not None:
            items.append(
                html.Li(
                    f"Estimated next dividend: {info['forecast_amount']:.2f} per share "
                    "(based on indicated annual dividend)"
                )
            )

        if info.get("annual_rate") is not None:
            items.append(
                html.Li(f"Indicated annual dividend: {info['annual_rate']:.2f} per share")
            )

        if info.get("yield") is not None:
            items.append(
                html.Li(
                    f"Indicated dividend yield: {info['yield'] * 100:.2f}%"
                )
            )

        if not items:
            info_block = html.P(
                "No upcoming dividend information available for this ticker.",
                style={"color": "#6b7280"},
            )
        else:
            info_block = html.Div(
                [
                    html.Strong("Upcoming dividend:"),
                    html.Ul(items),
                ]
            )

        return fig, info_block