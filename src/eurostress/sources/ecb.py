"""Point-in-time aware downloader for public ECB Data Portal series."""

from __future__ import annotations

from datetime import UTC, datetime
from io import StringIO
from urllib.parse import urlencode

import pandas as pd
import requests


class EcbClient:
    base_url = "https://data-api.ecb.europa.eu/service/data"

    def __init__(self, timeout_seconds: int = 30) -> None:
        self.timeout_seconds = timeout_seconds

    def build_url(self, flow: str, key: str, start: str, end: str) -> str:
        query = urlencode(
            {
                "startPeriod": start,
                "endPeriod": end,
                "format": "csvdata",
                "detail": "dataonly",
            }
        )
        return f"{self.base_url}/{flow}/{key}?{query}"

    def fetch(self, flow: str, key: str, start: str, end: str) -> pd.DataFrame:
        """Download one series and return normalized observations."""
        url = self.build_url(flow=flow, key=key, start=start, end=end)
        response = requests.get(
            url,
            timeout=self.timeout_seconds,
            headers={"User-Agent": "eurostress/0.1 research project"},
        )
        response.raise_for_status()

        frame = pd.read_csv(StringIO(response.text))
        required = {"TIME_PERIOD", "OBS_VALUE"}
        if not required <= set(frame.columns):
            raise ValueError(
                f"ECB response missing expected columns {sorted(required)}; "
                f"received {sorted(frame.columns)}"
            )

        observations = frame.loc[:, ["TIME_PERIOD", "OBS_VALUE"]].copy()
        observations.columns = ["date", "value"]
        observations["date"] = pd.to_datetime(observations["date"], errors="raise")
        observations["value"] = pd.to_numeric(observations["value"], errors="raise")
        observations["series_id"] = f"{flow}.{key}"
        observations["retrieved_at_utc"] = datetime.now(UTC)
        return observations.sort_values("date").drop_duplicates(["series_id", "date"])

