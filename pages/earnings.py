# pages/earnings.py
from dash import dcc, html, Input, Output, State
import pandas as pd

from trading.data_loader import YahooDataProvider


# -------------------------------------------------------------------
# Layout
# -------------------------------------------------------------------
def layout():
    """Page layout for the Earnings screen."""
    return html.Div(
        [
            html.H2("Earnings", style={"marginBottom": "16px"}),

            # Ticker input + button
            html.Div(
                [
                    dcc.Input(
                        id="earn-ticker",
                        type="text",
                        value="AAPL",
                        placeholder="Ticker",
                        style={"width": "140px", "marginRight": "8px"},
                    ),
                    html.Button(
                        "Load Earnings",
                        id="earn-load",
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

            # Main revenue/earnings table
            html.Div(id="earn-table", style={"marginBottom": "32px"}),

            # EPS surprises table
            html.Div(
                [
                    html.H3(
                        "Recent EPS surprises",
                        style={"marginBottom": "12px", "fontSize": "20px"},
                    ),
                    html.P(
                        "Compare analyst EPS estimates vs reported EPS. "
                        "We also show the next-session opening gap.",
                        style={
                            "fontSize": "13px",
                            "color": "#6b7280",
                            "marginBottom": "12px",
                        },
                    ),
                    html.Div(id="earn-surprise-table"),
                ]
            ),
        ]
    )


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def _choose_scale(df: pd.DataFrame) -> tuple[float, str]:
    """
    Choose an appropriate scale factor and unit label for Revenue/Earnings.

    Returns (scale_factor, unit_label), e.g. (1e9, "Bn"), (1e6, "M"), (1e3, "K"), (1, "")
    """
    numeric_series = []
    for col in df.columns:
        if "revenue" in col.lower() or "earning" in col.lower():
            numeric_series.append(pd.to_numeric(df[col], errors="coerce"))

    if not numeric_series:
        return 1.0, ""

    combined = pd.concat(numeric_series)
    max_val = combined.abs().max()

    if pd.isna(max_val) or max_val == 0:
        return 1.0, ""
    if max_val < 1e3:
        return 1.0, ""
    elif max_val < 1e6:
        return 1e3, "K"
    elif max_val < 1e9:
        return 1e6, "M"
    elif max_val < 1e12:
        return 1e9, "Bn"
    else:
        return 1e12, "T"


def _format_scaled(value, scale_factor: float) -> str:
    """Format a numeric value with the given scale factor, 2 decimals."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return ""
    return f"{v / scale_factor:,.2f}"


def _fmt_pct(x, decimals=2, show_sign=False):
    """Format a percentage value as text (no color here; color is applied in style)."""
    if pd.isna(x):
        return ""
    sign = "+" if (show_sign and x >= 0) else ""
    return f"{sign}{x:.{decimals}f}%"


# -------------------------------------------------------------------
# Callbacks
# -------------------------------------------------------------------
def register_callbacks(app, data_provider: YahooDataProvider):
    """
    Register callbacks for loading and displaying earnings.
    """

    @app.callback(
        [
            Output("earn-table", "children"),
            Output("earn-surprise-table", "children"),
        ],
        Input("earn-load", "n_clicks"),
        State("earn-ticker", "value"),
        prevent_initial_call=True,
    )
    def load_earnings(n_clicks, ticker):
        # ---------------------------------------------------------
        # 1) Revenue / Earnings table (auto-scaled units)
        # ---------------------------------------------------------
        df = data_provider.get_earnings(ticker)

        if df is None or df.empty:
            main_table = html.P("No earnings data available for this ticker.")
        else:
            # Normalize columns: Period, Revenue, Earnings, Type
            col_map = {}
            for c in df.columns:
                lower = c.lower()
                if "period" == lower:
                    col_map[c] = "Period"
                elif "revenue" in lower:
                    col_map[c] = "Revenue"
                elif "earning" in lower:
                    col_map[c] = "Earnings"
                elif "type" == lower:
                    col_map[c] = "Type"

            df = df.rename(columns=col_map)
            cols = [c for c in ["Period", "Revenue", "Earnings", "Type"] if c in df.columns]
            df = df[cols].copy()

            scale_factor, unit_label = _choose_scale(df)

            # Header
            header_cells = []
            for col in cols:
                if col in ("Revenue", "Earnings") and unit_label:
                    label = f"{col} ({unit_label})"
                else:
                    label = col

                header_cells.append(
                    html.Th(
                        label,
                        style={
                            "padding": "6px 12px",
                            "textAlign": "center",
                            "fontSize": "15px",
                        },
                    )
                )

            header = html.Tr(header_cells)

            # Rows
            body_rows = []
            for _, row in df.iterrows():
                row_cells = []
                for col in cols:
                    if col in ("Revenue", "Earnings"):
                        text = _format_scaled(row[col], scale_factor)
                    else:
                        text = str(row[col])

                    row_cells.append(
                        html.Td(
                            text,
                            style={
                                "padding": "6px 12px",
                                "textAlign": "center",
                                "fontSize": "13px",
                            },
                        )
                    )
                body_rows.append(html.Tr(row_cells))

            main_table = html.Table(
                [html.Thead(header), html.Tbody(body_rows)],
                style={
                    "borderCollapse": "collapse",
                    "width": "100%",
                    "marginTop": "12px",
                },
            )

        # ---------------------------------------------------------
        # 2) EPS surprises table (estimate vs actual vs surprise + gap)
        # ---------------------------------------------------------
        eps_df = data_provider.get_eps_surprises_with_gaps(
            ticker, limit=16, include_gap=True
        )

        if eps_df is None or eps_df.empty:
            surprise_table = html.P(
                "No EPS surprise data available for this ticker.",
                style={"fontSize": "13px", "color": "#6b7280"},
            )
        else:
            df_eps = eps_df.copy()

            # For display
            df_eps["Date"] = pd.to_datetime(df_eps["date"]).dt.strftime("%Y-%m-%d")
            df_eps["Period"] = df_eps["period"].astype(str)
            df_eps["Estimate EPS"] = pd.to_numeric(df_eps["eps_estimate"], errors="coerce").round(2)
            df_eps["Actual EPS"] = pd.to_numeric(df_eps["eps_actual"], errors="coerce").round(2)
            df_eps["Surprise (%)"] = pd.to_numeric(df_eps["surprise_pct"], errors="coerce").round(2)

            has_gap = "gap_next_open_pct" in df_eps.columns
            if has_gap:
                df_eps["Gap next open (%)"] = pd.to_numeric(
                    df_eps["gap_next_open_pct"], errors="coerce"
                ).round(2)

            def color_posneg(val):
                if pd.isna(val):
                    return "inherit"
                if val > 0:
                    return "#16a34a"  # green
                if val < 0:
                    return "#dc2626"  # red
                return "inherit"

            # Header
            header_cells = [
                html.Th("Earnings date", style={"textAlign": "left", "padding": "6px 12px"}),
                html.Th("Period", style={"textAlign": "left", "padding": "6px 12px"}),
                html.Th("Estimate EPS", style={"textAlign": "right", "padding": "6px 12px"}),
                html.Th("Actual EPS", style={"textAlign": "right", "padding": "6px 12px"}),
                html.Th("Surprise (%)", style={"textAlign": "right", "padding": "6px 12px"}),
            ]
            if has_gap:
                header_cells.append(
                    html.Th("Gap next open (%)", style={"textAlign": "right", "padding": "6px 12px"})
                )

            header_eps = html.Tr(header_cells)

            # Body
            body_eps_rows = []
            for _, r in df_eps.iterrows():
                row = [
                    html.Td(r["Date"], style={"textAlign": "left", "padding": "6px 12px"}),
                    html.Td(r["Period"], style={"textAlign": "left", "padding": "6px 12px"}),
                    html.Td(f"{r['Estimate EPS']:.2f}", style={"textAlign": "right", "padding": "6px 12px"}),
                    html.Td(f"{r['Actual EPS']:.2f}", style={"textAlign": "right", "padding": "6px 12px"}),
                    html.Td(
                        _fmt_pct(r["Surprise (%)"], 2, show_sign=True),
                        style={
                            "textAlign": "right",
                            "padding": "6px 12px",
                            "fontWeight": 600,
                            "color": color_posneg(r["Surprise (%)"]),
                        },
                    ),
                ]
                if has_gap:
                    row.append(
                        html.Td(
                            _fmt_pct(r.get("Gap next open (%)"), 2, show_sign=True),
                            style={
                                "textAlign": "right",
                                "padding": "6px 12px",
                                "color": color_posneg(r.get("Gap next open (%)")),
                            },
                        )
                    )
                body_eps_rows.append(html.Tr(row))

            surprise_table = html.Table(
                [html.Thead(header_eps), html.Tbody(body_eps_rows)],
                style={
                    "borderCollapse": "collapse",
                    "width": "100%",
                    "marginTop": "4px",
                    "fontSize": "14px",
                },
            )

        return main_table, surprise_table
