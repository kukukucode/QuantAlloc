"""Check ticker search ranking and market filtering without external requests."""

from datetime import date
from io import BytesIO
from types import SimpleNamespace

import pandas as pd

from src import symbol_search


def test_jpx_catalog_searches_names_and_alphanumeric_codes(monkeypatch) -> None:
    workbook = BytesIO()
    pd.DataFrame(
        {
            "銘柄名": [
                "豊田通商",
                "トヨタ自動車",
                "Ｖｅｒｉｔａｓ　Ｉｎ　Ｓｉｌｉｃｏ",
            ],
            "コード": [8015, 7203, "130A"],
        }
    ).to_excel(workbook, index=False)
    monkeypatch.setattr(
        symbol_search, "urlopen", lambda *args, **kwargs: BytesIO(workbook.getvalue())
    )

    symbol_search._jpx_listings.cache_clear()
    try:
        for query, symbol in [
            ("トヨタ", "7203.T"),
            ("７２０３．Ｔ", "7203.T"),
            ("130a", "130A.T"),
        ]:
            matches = symbol_search.search_japanese_symbols(
                symbol_search.normalize_symbol_query(query)
            )
            assert matches[0].symbol == symbol
        assert symbol_search.search_japanese_symbols(".t") == []
        assert symbol_search._jpx_listings.cache_info().hits == 3
    finally:
        symbol_search._jpx_listings.cache_clear()


def test_yahoo_search_excludes_other_markets_and_duplicate_quotes(monkeypatch) -> None:
    requests = []

    def search(query, **options):
        requests.append(query)
        return SimpleNamespace(
            quotes=[
                {"symbol": "BRKC", "exchange": "PCX", "quoteType": "ETF"},
                {"symbol": "BRK-B.TO", "exchange": "TOR", "quoteType": "EQUITY"},
                {
                    "symbol": "BRK-B",
                    "exchange": "NYQ",
                    "quoteType": "EQUITY",
                    "shortname": "Berkshire Hathaway Inc.",
                },
                {"symbol": "BRK-B", "exchange": "NYQ", "quoteType": "EQUITY"},
                {"symbol": "BTC-USD", "exchange": "CCC", "quoteType": "CRYPTOCURRENCY"},
            ]
        )

    monkeypatch.setattr(symbol_search.yf, "Search", search)
    symbol_search._search_us_quotes.cache_clear()
    try:
        matches = symbol_search.search_us_symbols("brk.b")
        assert [match.symbol for match in matches] == ["BRK-B", "BRKC"]
        assert matches[0].name == "Berkshire Hathaway Inc."
        assert symbol_search._search_us_quotes("BRK-B", date.today())
        assert requests == ["BRK-B"]
    finally:
        symbol_search._search_us_quotes.cache_clear()
