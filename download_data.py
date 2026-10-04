"""
Download the full daily price history of every ticker from Yahoo Finance.

    python download_data.py

Uses the `yfinance` package (unofficial Yahoo Finance client, no API key).
Each file holds every trading day from the listing date (or the earliest date
Yahoo provides) to the last completed session. Yahoo Finance data is for
personal and research use; see Yahoo's terms before any commercial use.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import yfinance as yf

TICKERS = ["AAPL", "NVDA", "TSLA", "SPY", "QQQ", "MSFT", "GOOGL", "AMZN", "META", "AMD", "JPM", "XOM"]
COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def main(out: Path = Path("data")) -> None:
    out.mkdir(exist_ok=True)
    summary = {"source": "Yahoo Finance via yfinance " + yf.__version__,
               "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "tickers": {}}
    for ticker in TICKERS:
        df = yf.download(ticker, period="max", interval="1d", auto_adjust=False, progress=False)
        df.columns = df.columns.get_level_values(0)
        df = df[COLUMNS].dropna()
        df.index.name = "Date"
        df.to_csv(out / f"{ticker}.csv", date_format="%Y-%m-%d")
        summary["tickers"][ticker] = {"rows": len(df), "first": str(df.index.min().date()), "last": str(df.index.max().date())}
        print(ticker, summary["tickers"][ticker])
    (out / "download_meta.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
