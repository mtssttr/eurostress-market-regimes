from urllib.parse import parse_qs, urlparse

from eurostress.sources.ecb import EcbClient


def test_ecb_url_contains_frozen_window_and_csv_format() -> None:
    url = EcbClient().build_url(
        flow="EXR",
        key="D.USD.EUR.SP00.A",
        start="2005-09-01",
        end="2025-12-31",
    )
    parsed = urlparse(url)
    query = parse_qs(parsed.query)

    assert parsed.path.endswith("/EXR/D.USD.EUR.SP00.A")
    assert query["startPeriod"] == ["2005-09-01"]
    assert query["endPeriod"] == ["2025-12-31"]
    assert query["format"] == ["csvdata"]
    assert query["detail"] == ["dataonly"]

