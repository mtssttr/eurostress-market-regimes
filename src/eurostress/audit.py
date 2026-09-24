"""Deterministic coverage checks for downloaded raw observations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from eurostress.config import ProjectConfig


@dataclass(frozen=True)
class CoverageResult:
    series_name: str
    daily_rows: int
    observation_start: str
    observation_end: str
    duplicate_dates: int
    null_values: int
    expected_weeks: int
    valid_weeks: int
    stale_weeks: int
    weekly_coverage_percent: float


def summarize_weekly_coverage(
    series_name: str,
    frame: pd.DataFrame,
    start: str,
    end: str,
    max_staleness_days: int,
) -> CoverageResult:
    required = {"series_id", "date", "value"}
    if not required <= set(frame.columns):
        raise ValueError(f"{series_name} is missing columns: {sorted(required - set(frame.columns))}")

    observations = frame.loc[:, ["series_id", "date", "value"]].copy()
    observations["date"] = pd.to_datetime(observations["date"], errors="raise").dt.tz_localize(
        None
    )
    observations = observations.sort_values("date")

    duplicate_dates = int(observations.duplicated(["series_id", "date"]).sum())
    null_values = int(observations["value"].isna().sum())
    if observations.empty:
        raise ValueError(f"{series_name} has no observations")

    indexed = observations.set_index("date")
    indexed["source_date"] = indexed.index
    weekly = indexed.loc[:, ["value", "source_date"]].resample("W-FRI").last()
    expected_fridays = pd.date_range(start=start, end=end, freq="W-FRI")
    weekly = weekly.reindex(expected_fridays)
    staleness = (weekly.index.to_series(index=weekly.index) - weekly["source_date"]).dt.days
    stale = weekly["value"].notna() & (staleness > max_staleness_days)
    valid = weekly["value"].notna() & ~stale

    return CoverageResult(
        series_name=series_name,
        daily_rows=len(observations),
        observation_start=observations["date"].min().date().isoformat(),
        observation_end=observations["date"].max().date().isoformat(),
        duplicate_dates=duplicate_dates,
        null_values=null_values,
        expected_weeks=len(expected_fridays),
        valid_weeks=int(valid.sum()),
        stale_weeks=int(stale.sum()),
        weekly_coverage_percent=round(100 * float(valid.mean()), 2),
    )


def generate_coverage_report(
    config: ProjectConfig,
    input_directory: str | Path,
    output_path: str | Path,
) -> list[CoverageResult]:
    input_path = Path(input_directory)
    start = config.project["start_date"]
    end = config.project["end_date"]
    max_staleness = int(config.project["max_staleness_calendar_days"])

    results: list[CoverageResult] = []
    for series in config.series:
        frame = pd.read_parquet(input_path / f"{series.name}.parquet")
        results.append(
            summarize_weekly_coverage(
                series_name=series.name,
                frame=frame,
                start=start,
                end=end,
                max_staleness_days=max_staleness,
            )
        )

    lines = [
        "# Raw-data coverage report",
        "",
        f"Frozen window: {start} to {end}. Maximum permitted staleness: {max_staleness} days.",
        "",
        "| Series | Daily rows | First date | Last date | Duplicate dates | Null values | Valid weeks | Stale weeks | Weekly coverage |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        values = asdict(result)
        lines.append(
            f"| {values['series_name']} | {values['daily_rows']:,} | "
            f"{values['observation_start']} | {values['observation_end']} | "
            f"{values['duplicate_dates']} | {values['null_values']} | "
            f"{values['valid_weeks']:,}/{values['expected_weeks']:,} | "
            f"{values['stale_weeks']} | {values['weekly_coverage_percent']:.2f}% |"
        )

    failures = [
        result
        for result in results
        if result.duplicate_dates > 0
        or result.null_values > 0
        or result.weekly_coverage_percent < 95
    ]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            (
                "PASS: every configured series meets the pre-committed 95% weekly-coverage "
                "threshold with no duplicate dates or null values."
                if not failures
                else "FAIL: at least one series violates a pre-committed coverage or integrity rule."
            ),
            "",
            (
                "This report checks availability and integrity only. It does not inspect returns, "
                "regimes, correlations, stress dates, or test-period model outcomes."
            ),
            "",
        ]
    )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")
    return results
