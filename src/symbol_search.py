"""Search listings by name or ticker in the selected market."""

import re
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from io import BytesIO
from unicodedata import normalize
from urllib.request import urlopen

import pandas as pd
import yfinance as yf

JP_TICKER = re.compile(r"[0-9][0-9A-Z][0-9][0-9A-Z]")
US_TICKER = re.compile(r"[A-Z][A-Z0-9]{0,5}(?:-[A-Z])?")
JPX_LISTINGS_URL = (
    "https://www.jpx.co.jp/markets/statistics-equities/misc/"
    "tvdivq0000001vg2-att/data_j.xlsx"
)
US_EXCHANGES = {"NYQ", "NMS", "NGM", "NCM", "ASE", "PCX", "BTS", "PNK", "OQB", "OQX"}
MAX_RESULTS = 8


@dataclass(frozen=True)
class SymbolMatch:
    symbol: str
    name: str


def normalize_symbol_query(query: str) -> str:
    return normalize("NFKC", query).strip().casefold()


@lru_cache(maxsize=1)
def _jpx_listings(day: date) -> tuple[SymbolMatch, ...]:
    with urlopen(JPX_LISTINGS_URL, timeout=8) as response:
        workbook = BytesIO(response.read(2_000_001))
    if workbook.getbuffer().nbytes > 2_000_000:
        raise ValueError("JPX listing file is too large")
    columns = ["コード", "銘柄名"]
    listings = pd.read_excel(workbook, usecols=columns, dtype=str)
    matches = tuple(
        SymbolMatch(f"{code}.T", name.strip())
        for code, name in listings[columns].dropna().itertuples(index=False, name=None)
        if JP_TICKER.fullmatch(code) and name.strip()
    )
    if not matches:
        raise ValueError("JPX listing file is empty")
    return matches


def search_japanese_symbols(query: str) -> list[SymbolMatch]:
    code_query = query.removesuffix(".t").upper()
    ranked: list[tuple[int, SymbolMatch]] = []
    for match in _jpx_listings(date.today()):
        code = match.symbol.removesuffix(".T")
        name = normalize_symbol_query(match.name)
        if code == code_query or name == query:
            ranked.append((0, match))
        elif (code_query and code.startswith(code_query)) or name.startswith(query):
            ranked.append((1, match))
        elif query in name:
            ranked.append((2, match))
    return [
        match for _, match in sorted(ranked, key=lambda item: item[0])[:MAX_RESULTS]
    ]


@lru_cache(maxsize=256)
def _search_us_quotes(query: str, day: date) -> tuple[SymbolMatch, ...]:
    response = yf.Search(
        query,
        max_results=24,
        news_count=0,
        lists_count=0,
        recommended=0,
        include_cb=False,
        timeout=8,
    )
    matches: dict[str, SymbolMatch] = {}
    for quote in response.quotes:
        symbol = quote.get("symbol", "")
        if (
            isinstance(symbol, str)
            and quote.get("exchange") in US_EXCHANGES
            and quote.get("quoteType") in {"EQUITY", "ETF"}
            and US_TICKER.fullmatch(symbol)
        ):
            name = quote.get("shortname") or quote.get("longname") or symbol
            matches.setdefault(symbol, SymbolMatch(symbol, name))
    return tuple(matches.values())


def search_us_symbols(query: str) -> list[SymbolMatch]:
    symbol = query.upper().replace(".", "-")
    keyword = symbol if US_TICKER.fullmatch(symbol) else query
    matches = _search_us_quotes(keyword, date.today())
    return sorted(matches, key=lambda match: match.symbol != symbol)[:MAX_RESULTS]
