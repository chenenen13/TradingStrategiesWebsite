from dash import html


def layout():
    return html.Div(
        [
            html.H2("Resources", style={"marginBottom": "16px"}),
            html.Ul(
                [
                    html.Li(
                        html.A(
                            "Yahoo Finance",
                            href="https://finance.yahoo.com",
                            target="_blank",
                        )
                    ),
                    html.Li(
                        html.A(
                            "yfinance documentation",
                            href="https://pypi.org/project/yfinance/",
                            target="_blank",
                        )
                    ),
                    html.Li(
                        html.A(
                            "Quantitative Finance StackExchange",
                            href="https://quant.stackexchange.com",
                            target="_blank",
                        )
                    ),
                    html.Li(
                        html.A(
                            "Investopedia - Technical Analysis",
                            href=(
                                "https://www.investopedia.com/"
                                "technical-analysis-4689657"
                            ),
                            target="_blank",
                        )
                    ),
                ]
            ),
        ]
    )


def register_callbacks(app, *args, **kwargs):
    # No callbacks for this page, but keep the function for consistency
    return
