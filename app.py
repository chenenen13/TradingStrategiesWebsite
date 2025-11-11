from dash import Dash, dcc, html, Input, Output

from trading.data_loader import YahooDataProvider
from trading.backtesting import Backtester

# import page modules
from pages.strategies import layout as strategies_layout, register_callbacks as register_strategies
from pages.outliers import layout as outliers_layout, register_callbacks as register_outliers
from pages import outliers
from pages.news import layout as news_layout, register_callbacks as register_news
from pages.dividends import layout as dividends_layout, register_callbacks as register_dividends
from pages.earnings import layout as earnings_layout, register_callbacks as register_earnings
from pages.resources import layout as resources_layout

import os
from dotenv import load_dotenv

load_dotenv()  # charge .env
NEWS_API_KEY = os.getenv("78968501fb8942bc828b97f2d06ededc")


# -------------------------------------------------------------------
# App + services
# -------------------------------------------------------------------
app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server  # for Render / gunicorn

data_provider = YahooDataProvider()
backtester = Backtester()

# -------------------------------------------------------------------
# Global layout: sidebar + main content
# -------------------------------------------------------------------
app.layout = html.Div(
    style={
        "display": "flex",
        "minHeight": "100vh",
        "fontFamily": "system-ui, -apple-system, BlinkMacSystemFont, sans-serif",
        "backgroundColor": "#e5e7eb",
    },
    children=[
        dcc.Location(id="url"),
        # Sidebar
        html.Div(
            style={
                "width": "240px",
                "background": "linear-gradient(180deg, #0f172a, #1e293b)",
                "color": "white",
                "padding": "24px 16px",
                "display": "flex",
                "flexDirection": "column",
                "gap": "16px",
            },
            children=[
                html.Div(
                    [
                        html.Div("📊", style={"fontSize": "26px"}),
                        html.H2(
                            "Trading Lab",
                            style={
                                "fontSize": "30px",
                                "marginBottom": "0",
                                "marginTop": "6px",
                            },
                        ),
                        html.Div(
                            "Backtesting playground",
                            style={"fontSize": "12px", "color": "#9ca3af"},
                        ),
                    ]
                ),
                html.Hr(style={"borderColor": "#4b5563"}),

                html.Div(
                    [
                        html.P(
                            "Navigation",
                            style={
                                "textTransform": "uppercase",
                                "letterSpacing": "0.08em",
                                "color": "#9ca3af",
                                "fontSize": "18px",
                                "fontWeight": 500,
                            },
                        ),
                        dcc.Link(
                            "📈  Strategies",
                            href="/strategies",
                            style={
                                "display": "block",
                                "padding": "8px 10px",
                                "borderRadius": "10px",
                                "color": "white",
                                "textDecoration": "none",
                                "fontSize": "16px",  
                                "fontWeight": 500,     
                            },
                        ),
                        dcc.Link(
                            "📉  Outliers",
                            href="/outliers",
                            style={
                                "display": "block",
                                "padding": "8px 10px",
                                "borderRadius": "10px",
                                "color": "white",
                                "textDecoration": "none",
                                "fontSize": "16px",  
                                "fontWeight": 500,
                            },
                        ),
                        dcc.Link(
                            "📰  News",
                            href="/news",
                            style={
                                "display": "block",
                                "padding": "8px 10px",
                                "borderRadius": "10px",
                                "color": "white",
                                "textDecoration": "none",
                                "fontSize": "16px",
                                "fontWeight": 500,
                            },
                        ),
                        dcc.Link(
                            "💸  Dividends",
                            href="/dividends",
                            style={
                                "display": "block",
                                "padding": "8px 10px",
                                "borderRadius": "10px",
                                "color": "white",
                                "textDecoration": "none",
                                "fontSize": "16px",
                                "fontWeight": 500,
                            },
                        ),
                        dcc.Link(
                            "📅  Earnings",
                            href="/earnings",
                            style={
                                "display": "block",
                                "padding": "8px 10px",
                                "borderRadius": "10px",
                                "color": "white",
                                "textDecoration": "none",
                                "fontSize": "16px",
                                "fontWeight": 500,
                            },
                        ),
                        dcc.Link(
                            "📚  Resources",
                            href="/resources",
                            style={
                                "display": "block",
                                "padding": "8px 10px",
                                "borderRadius": "10px",
                                "color": "white",
                                "textDecoration": "none",
                                "fontSize": "16px",
                                "fontWeight": 500,
                            },
                        ),
                    ]
                ),
            ],
        ),
        # Main content
        html.Div(
            id="page-content",
            style={
                "flex": "1",
                "padding": "28px 32px",
                "maxWidth": "1400px",
                "margin": "0 auto",
            },
        ),
    ],
)

# -------------------------------------------------------------------
# Routing callback
# -------------------------------------------------------------------
@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def display_page(pathname):
    if pathname in ("/", None, "/strategies"):
        return strategies_layout()
    elif pathname == "/outliers":
        return outliers.layout()
    elif pathname == "/news":
        return news_layout()
    elif pathname == "/dividends":
        return dividends_layout()
    elif pathname == "/earnings":
        return earnings_layout()
    elif pathname == "/resources":
        return resources_layout()
    else:
        return html.Div([html.H2("404 - Page not found")])


# -------------------------------------------------------------------
# Register callbacks from each page module
# -------------------------------------------------------------------
register_strategies(app, data_provider, backtester)
register_outliers(app, data_provider)
register_news(app, data_provider)
register_dividends(app, data_provider)
register_earnings(app, data_provider)
# resources page has no callbacks

# -------------------------------------------------------------------
# Run local
# -------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
