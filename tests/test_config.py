from eurostress.config import load_config
from eurostress.sources.yahoo import YahooClient


def test_frozen_split_and_series_are_present() -> None:
    config = load_config("config/series.yml")

    assert config.project["start_date"] == "2007-03-30"
    assert config.project["end_date"] == "2025-12-31"
    assert config.splits["test_start"] == "2020-01-01"
    assert config.evaluation["ewma_lambda"] == 0.94
    assert config.evaluation["target_horizon_calendar_days"] == 7
    assert config.models["state_counts"] == [2, 3, 4]
    assert config.models["deterministic_initializations"] == 20
    assert {series.name for series in config.series} == {
        "euro_stoxx_50",
        "eurusd",
        "euro_aaa_2y",
        "euro_aaa_10y",
        "euro_all_ratings_10y",
        "new_ciss",
    }


def test_external_validation_is_not_a_model_input() -> None:
    config = load_config("config/series.yml")
    ciss = next(series for series in config.series if series.name == "new_ciss")

    assert ciss.role == "external_validation_only"


def test_yahoo_end_date_is_made_inclusive() -> None:
    assert YahooClient.exclusive_end_date("2025-12-31") == "2026-01-01"
