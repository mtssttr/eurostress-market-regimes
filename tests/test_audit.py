import pandas as pd

from eurostress.audit import summarize_weekly_coverage


def test_weekly_coverage_rejects_observations_older_than_limit() -> None:
    frame = pd.DataFrame(
        {
            "series_id": ["TEST", "TEST"],
            "date": ["2024-01-05", "2024-01-06"],
            "value": [1.0, 2.0],
        }
    )

    result = summarize_weekly_coverage(
        series_name="test",
        frame=frame,
        start="2024-01-05",
        end="2024-01-12",
        max_staleness_days=5,
    )

    assert result.expected_weeks == 2
    assert result.valid_weeks == 1
    assert result.stale_weeks == 1
    assert result.weekly_coverage_percent == 50.0

