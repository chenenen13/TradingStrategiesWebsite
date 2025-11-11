from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import pandas as pd

from trading.data_loader import YahooDataProvider
# from trading.outliers import OutlierDetector  # optional, not needed in this version


# -------------------------------------------------------------------
# Layout
# -------------------------------------------------------------------
def layout():
    """Page layout for the Outlier Detection screen."""
    return html.Div(
        [
            html.H2("Outlier Detection", style={"marginBottom": "16px"}),

            # Input controls
            html.Div(
                style={
                    "display": "flex",
                    "gap": "8px",
                    "marginBottom": "16px",
                    "flexWrap": "wrap",
                },
                children=[
                    dcc.Input(
                        id="out-ticker",
                        type="text",
                        value="AAPL",
                        placeholder="Ticker",
                        style={"width": "120px"},
                    ),
                    dcc.Input(
                        id="out-start-date",
                        type="text",
                        value="2020-01-01",
                        placeholder="Start (YYYY-MM-DD)",
                        style={"width": "150px"},
                    ),
                    dcc.Input(
                        id="out-end-date",
                        type="text",
                        value="2024-01-01",
                        placeholder="End (YYYY-MM-DD)",
                        style={"width": "150px"},
                    ),
                    dcc.Input(
                        id="out-window",
                        type="number",
                        value=20,
                        placeholder="Rolling window",
                        style={"width": "130px"},
                    ),
                    dcc.Input(
                        id="out-z",
                        type="number",
                        value=3.0,
                        placeholder="Z-score threshold",
                        style={"width": "150px", "step": "0.1"},
                    ),
                    html.Button(
                        "Detect Outliers",
                        id="run-outliers",
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

            # Summary text
            html.Div(id="outliers-stats", style={"marginBottom": "12px"}),

            # Price + outliers chart
            dcc.Graph(id="outliers-price-graph", style={"height": "500px"}),

            # Detailed table with each outlier
            html.Div(id="outliers-table", style={"marginTop": "20px"}),
        ]
    )


# -------------------------------------------------------------------
# Callbacks
# -------------------------------------------------------------------
def register_callbacks(app, data_provider: YahooDataProvider):
    """
    Register the Dash callbacks for the Outlier page.

    data_provider is shared at app level (YahooDataProvider instance).
    """

    @app.callback(
        [
            Output("outliers-price-graph", "figure"),
            Output("outliers-stats", "children"),
            Output("outliers-table", "children"),
        ],
        Input("run-outliers", "n_clicks"),
        [
            State("out-ticker", "value"),
            State("out-start-date", "value"),
            State("out-end-date", "value"),
            State("out-window", "value"),
            State("out-z", "value"),
        ],
        prevent_initial_call=True,
    )
    def run_outliers(n_clicks, ticker, start, end, window, z_thr):
        """
        1) Load price series
        2) Compute rolling mean / std and z-score
        3) Flag outliers where |z| > threshold
        4) Return chart + summary + detailed table
        """

        # 1) Load price data
        price_series = data_provider.get_price_series(ticker, start, end)
        df = price_series.data

        if df.empty:
            fig_empty = go.Figure()
            fig_empty.update_layout(
                title="No data available for this period",
                margin=dict(l=40, r=20, t=40, b=30),
            )
            return fig_empty, html.P("No data available."), html.Div()

        # 2) Compute rolling statistics and z-score
        window = int(window)
        z_thr = float(z_thr)

        df = df.copy()
        df["MA"] = df["price"].rolling(window).mean()
        df["STD"] = df["price"].rolling(window).std()
        df["zscore"] = (df["price"] - df["MA"]) / df["STD"]

        # 3) Detect outliers
        df["outlier"] = df["zscore"].abs() > z_thr
        outliers_df = df[df["outlier"]]
        nb_outliers = len(outliers_df)

        # 4) Build price + outliers chart
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["price"],
                name="Price",
                mode="lines",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=outliers_df.index,
                y=outliers_df["price"],
                mode="markers",
                name="Outliers",
                marker=dict(size=9, symbol="circle-open", color="red"),
            )
        )

        fig.update_layout(
            margin=dict(l=40, r=20, t=40, b=30),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
        )

        # 5) Summary text (number of outliers + last date)
        if nb_outliers > 0:
            last_outlier = outliers_df.index.max().strftime("%Y-%m-%d")
            stats_html = html.Ul(
                [
                    html.Li(f"Number of outliers detected: {nb_outliers}"),
                    html.Li(f"Last outlier date: {last_outlier}"),
                ]
            )
        else:
            stats_html = html.P("No outliers detected.")

        # 6) Detailed table with date, price, z-score and spread
        if nb_outliers > 0:
            details = outliers_df.copy()

            details["Date"] = details.index.strftime("%Y-%m-%d")
            details["Price"] = details["price"].round(2)
            details["Z-Score"] = details["zscore"].round(2)
            # Spread (%) = distance from rolling mean in percentage
            details["Spread (%)"] = (
                (details["price"] - details["MA"]) / details["MA"] * 100
            ).round(2)

            # Common cell style
            cell_style = {
                "padding": "6px 10px",
                "textAlign": "center",
                "fontSize": "13px",
            }

            # Header: 4 columns, same order as the data below
            header = html.Tr(
                [
                    html.Th("Date", style=cell_style),
                    html.Th("Price", style=cell_style),
                    html.Th("Z-Score", style=cell_style),
                    html.Th("Spread (%)", style=cell_style),
                ]
            )

            # Body rows
            rows = []
            for _, row in details.iterrows():
                spread_color = "#16a34a" if row["Spread (%)"] >= 0 else "#dc2626"
                rows.append(
                    html.Tr(
                        [
                            html.Td(row["Date"], style=cell_style),
                            html.Td(f"{row['Price']:.2f}", style=cell_style),
                            html.Td(f"{row['Z-Score']:.2f}", style=cell_style),
                            html.Td(
                                f"{row['Spread (%)']:.2f}%",
                                style={**cell_style, "color": spread_color},
                            ),
                        ]
                    )
                )

            table = html.Table(
                [html.Thead(header), html.Tbody(rows)],
                style={
                    "borderCollapse": "collapse",
                    "width": "100%",
                    "marginTop": "8px",
                },
            )
        else:
            table = html.Div()


        return fig, stats_html, table
