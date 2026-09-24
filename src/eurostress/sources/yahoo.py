"""Replaceable Yahoo Finance adapter for the prototype equity series."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pandas as pd
import yfinance as yf


class YahooClient:
    @staticmethod
    def exclusive_end_date(end: str) -> str:
        """Convert the project's inclusive end date to Yahoo's exclusive bound."""
        return (date.fromisoformat(end) + timedelta(days=1)).isoformat()

    def fetch(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Download daily closes without redistributing the provider's raw data."""
        raw = yf.download(
            symbol,
            start=start,
            end=self.exclusive_end_date(end),
            auto_adjust=False,
            actions=False,
            progress=False,
            threads=False,
        )
        if raw.empty:
            raise ValueError(f"Yahoo returned no observations for {symbol!r}")

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        value_column = "Adj Close" if "Adj Close" in raw.columns else "Close"
        observations = raw.loc[:, [value_column]].reset_index()
        observations.columns = ["date", "value"]
        observations["date"] = pd.to_datetime(observations["date"], errors="raise")
        observations["value"] = pd.to_numeric(observations["value"], errors="raise")
        observations["series_id"] = f"YAHOO.{symbol}"
        observations["retrieved_at_utc"] = datetime.now(UTC)
        return observations.sort_values("date").drop_duplicates(["series_id", "date"])
