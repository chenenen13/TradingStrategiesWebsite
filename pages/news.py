import os
import requests
from dash import dcc, html, Input, Output, State, no_update
from trading.data_loader import YahooDataProvider

NEWS_API_ENDPOINT = "https://newsapi.org/v2/top-headlines"


# -------------------------------------------------------------------
# Layout
# -------------------------------------------------------------------
def layout():
    return html.Div(
        [
            html.H2("Latest Financial News", style={"marginBottom": "24px"}),

            # Row for index "pills"
            html.Div(id="indices-row", style={"marginBottom": "24px"}),

            # Search bar
            html.Div(
                style={
                    "backgroundColor": "white",
                    "padding": "16px 20px",
                    "borderRadius": "16px",
                    "boxShadow": "0 18px 40px rgba(15,23,42,0.08)",
                    "display": "flex",
                    "gap": "10px",
                    "alignItems": "center",
                    "marginBottom": "24px",
                },
                children=[
                    dcc.Input(
                        id="news-query",
                        type="text",
                        value="markets",
                        placeholder="Search topics (e.g. stocks, tech, rates)",
                        style={
                            "flex": "1",
                            "height": "40px",
                            "borderRadius": "999px",
                            "border": "1px solid #e5e7eb",
                            "padding": "0 14px",
                            "fontSize": "14px",
                        },
                    ),
                    html.Button(
                        "Search",
                        id="news-search-btn",
                        style={
                            "height": "40px",
                            "padding": "0 24px",
                            "borderRadius": "999px",
                            "border": "none",
                            "backgroundColor": "#2563eb",
                            "color": "white",
                            "fontWeight": 500,
                            "cursor": "pointer",
                        },
                    ),
                ],
            ),

            # Grid of news cards
            html.Div(
                id="news-grid",
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(320px, 1fr))",
                    "gap": "20px",
                },
            ),
        ]
    )


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def _fetch_top_headlines(query: str | None = None):
    """
    Fetch financial headlines from NewsAPI.
    If query is provided, use that as 'q'. Otherwise, general business news.
    """
    api_key = "78968501fb8942bc828b97f2d06ededc"
    if not api_key:
        return None, "NEWS_API_KEY environment variable is not set."

    params = {
        "apiKey": api_key,
        "language": "en",
        "pageSize": 12,
    }

    # If user provided a search query, use everything endpoint
    if query and query.strip():
        endpoint = "https://newsapi.org/v2/everything"
        params["q"] = query.strip()
        params["sortBy"] = "publishedAt"
    else:
        endpoint = NEWS_API_ENDPOINT
        params["country"] = "us"
        params["category"] = "business"

    try:
        resp = requests.get(endpoint, params=params, timeout=6)
    except Exception as e:
        return None, f"Error calling NewsAPI: {e}"

    if resp.status_code != 200:
        return None, f"NewsAPI error: HTTP {resp.status_code}"

    data = resp.json()
    return data.get("articles", []), None


def _build_indices_row(indices_snapshot):
    if not indices_snapshot:
        return html.P("Could not load index data.", style={"color": "#6b7280"})

    cards = []
    for row in indices_snapshot:
        color = "#16a34a" if row["change_pct"] >= 0 else "#dc2626"
        sign = "+" if row["change_pct"] >= 0 else ""
        cards.append(
            html.Div(
                style={
                    "minWidth": "150px",
                    "padding": "10px 14px",
                    "borderRadius": "999px",
                    "backgroundColor": "white",
                    "boxShadow": "0 12px 30px rgba(15,23,42,0.06)",
                    "display": "flex",
                    "flexDirection": "column",
                    "gap": "2px",
                },
                children=[
                    html.Div(
                        row["name"],
                        style={"fontSize": "13px", "fontWeight": 600},
                    ),
                    html.Div(
                        f"{row['ticker']}",
                        style={"fontSize": "11px", "color": "#6b7280"},
                    ),
                    html.Div(
                        [
                            html.Span(
                                f"{row['last']:.2f}",
                                style={"fontSize": "13px", "marginRight": "8px"},
                            ),
                            html.Span(
                                f"{sign}{row['change_pct']:.2f}%",
                                style={"fontSize": "13px", "color": color},
                            ),
                        ]
                    ),
                ],
            )
        )

    return html.Div(
        cards,
        style={
            "display": "flex",
            "flexWrap": "wrap",
            "gap": "12px",
        },
    )


def _build_news_cards(articles, error_msg: str | None = None):
    if error_msg:
        return html.P(error_msg, style={"color": "#dc2626"})

    if not articles:
        return html.P("No articles found for this query.", style={"color": "#6b7280"})

    cards = []
    for a in articles:
        title = a.get("title", "No title")
        url = a.get("url", "#")
        source = a.get("source", {}).get("name", "")
        desc = a.get("description") or ""
        image = a.get("urlToImage")

        # small trimming of description
        if len(desc) > 180:
            desc = desc[:177] + "..."

        cards.append(
            html.Div(
                style={
                    "backgroundColor": "white",
                    "borderRadius": "18px",
                    "boxShadow": "0 18px 40px rgba(15,23,42,0.08)",
                    "overflow": "hidden",
                    "display": "flex",
                    "flexDirection": "column",
                    "height": "100%",
                },
                children=[
                    # image (optional)
                    html.Div(
                        style={"height": "160px", "overflow": "hidden"}
                        if image
                        else {"display": "none"},
                        children=(
                            html.Img(
                                src=image,
                                style={
                                    "width": "100%",
                                    "height": "100%",
                                    "objectFit": "cover",
                                },
                            )
                            if image
                            else None
                        ),
                    ),
                    html.Div(
                        style={"padding": "16px 18px", "display": "flex", "flexDirection": "column", "gap": "6px", "flex": "1"},
                        children=[
                            html.Div(
                                title,
                                style={
                                    "fontWeight": 600,
                                    "fontSize": "15px",
                                    "lineHeight": "1.3",
                                },
                            ),
                            html.Div(
                                desc,
                                style={
                                    "fontSize": "13px",
                                    "color": "#4b5563",
                                },
                            ),
                            html.Div(
                                source,
                                style={
                                    "fontSize": "12px",
                                    "color": "#9ca3af",
                                    "marginTop": "4px",
                                },
                            ),
                            html.Div(style={"flex": "1"}),  # spacer
                            html.A(
                                "Read more",
                                href=url,
                                target="_blank",
                                style={
                                    "alignSelf": "flex-start",
                                    "marginTop": "8px",
                                    "padding": "6px 12px",
                                    "borderRadius": "999px",
                                    "border": "1px solid #2563eb",
                                    "color": "#2563eb",
                                    "fontSize": "13px",
                                    "textDecoration": "none",
                                    "fontWeight": 500,
                                },
                            ),
                        ],
                    ),
                ],
            )
        )

    return cards


# -------------------------------------------------------------------
# Callbacks
# -------------------------------------------------------------------
def register_callbacks(app, data_provider: YahooDataProvider):

    # 1) indexes loaded automatically on page load
    @app.callback(
        Output("indices-row", "children"),
        Input("url", "pathname"),
    )
    def init_indices(pathname):
        # We only do something on the /news page
        if pathname != "/news":
            return no_update

        indices_snapshot = data_provider.get_global_indices_snapshot()
        return _build_indices_row(indices_snapshot)

    # 2) news search
    @app.callback(
        Output("news-grid", "children"),
        Input("news-search-btn", "n_clicks"),
        State("news-query", "value"),
        prevent_initial_call=True,
    )
    def load_news(n_clicks, query):
        articles, err = _fetch_top_headlines(query)
        news_cards = _build_news_cards(articles, err)
        return news_cards

