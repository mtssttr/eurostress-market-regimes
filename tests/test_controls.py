import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from eurostress import cli
from eurostress.config import ProjectConfig, SeriesConfig
from eurostress.controls import DataQualityError, enforce_raw_controls, inspect_frame
from eurostress.demo import run_demo


def check(frame, unit="index_points"):
    return inspect_frame(
        frame, series="equity", as_of="2024-01-12", expected_unit="index_points", declared_unit=unit
    )


def test_controls_detect_corruptions_and_do_not_guess_missing_units():
    frame = pd.DataFrame({"date": ["2024-01-11", "2024-01-12"], "value": [100.0, 101.0]})
    assert check(frame) == []
    assert check(frame, None)[0].severity == "WARNING"
    assert {x.rule for x in check(pd.concat([frame, frame.iloc[[0]]]))} == {"duplicate_date"}
    frame.loc[0, "value"] = np.inf
    assert "missing_or_nonfinite" in {x.rule for x in check(frame)}
    frame.loc[0, "date"] = "2024-01-15"
    assert "future_observation" in {x.rule for x in check(frame)}
    assert "unit_mismatch" in {x.rule for x in check(frame, "percent_per_annum")}


def test_failed_coverage_command_has_nonzero_exit(monkeypatch):
    monkeypatch.setattr("sys.argv", ["eurostress", "coverage"])
    monkeypatch.setattr(cli, "load_config", lambda *_: None)
    monkeypatch.setattr(
        cli,
        "generate_coverage_report",
        lambda *_: [
            SimpleNamespace(
                series_name="broken", weekly_coverage_percent=50, duplicate_dates=0, null_values=0
            )
        ],
    )
    assert cli.main() == 1


def test_manifest_mismatch_blocks_before_calculation(tmp_path):
    source = SeriesConfig("equity", "yahoo", "input", "test", "index_points", "prohibited", ())
    config = ProjectConfig(
        {"start_date": "2024-01-05", "end_date": "2024-01-12", "max_staleness_calendar_days": 5},
        {},
        (source,),
    )
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-05", "2024-01-12"]),
            "value": [100.0, 101.0],
            "series_id": ["TEST", "TEST"],
        }
    )
    path = tmp_path / "equity.parquet"
    frame.to_parquet(path)
    entry = {
        "series_name": "equity",
        "unit": "index_points",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    (tmp_path / "manifest.json").write_text(json.dumps({"series": [entry]}))
    report = tmp_path / "report.json"
    assert enforce_raw_controls(config, tmp_path, report)["status"] == "PASS"
    frame.loc[1, "value"] = 102.0
    frame.to_parquet(path)
    with pytest.raises(DataQualityError):
        enforce_raw_controls(config, tmp_path, report)
    assert "source_hash" in {x["rule"] for x in json.loads(report.read_text())["findings"]}


def test_offline_demo_blocks_each_case_then_recomputes(tmp_path):
    result = run_demo(tmp_path)
    assert len(result["cases"]) == 4
    assert all(x["before"] == "BLOCKED" and x["after"] == "PASS" for x in result["cases"])
    assert all(x["calculation_before"] == "withheld" for x in result["cases"])
    assert pd.read_csv(tmp_path / "synthetic_source.csv").shape[0] > 50


def test_failed_gate_prevents_feature_output(tmp_path, monkeypatch):
    from eurostress.features import build_feature_dataset

    monkeypatch.chdir(tmp_path)
    raw = tmp_path / "raw"
    raw.mkdir()
    output = tmp_path / "processed"
    source = SeriesConfig("equity", "yahoo", "input", "test", "index_points", "prohibited", ())
    config = ProjectConfig(
        {"start_date": "2024-01-05", "end_date": "2024-01-12", "max_staleness_calendar_days": 5},
        {},
        (source,),
    )
    with pytest.raises(DataQualityError):
        build_feature_dataset(config, "unused.yml", raw, output, "unused.duckdb")
    assert not output.exists()
    assert not (tmp_path / "unused.duckdb").exists()
    assert (
        json.loads((tmp_path / "reports/controls/data_quality.json").read_text())["status"]
        == "BLOCKED"
    )


def test_missing_identifier_produces_a_blocking_report(tmp_path):
    source = SeriesConfig("equity", "yahoo", "input", "test", "index_points", "prohibited", ())
    config = ProjectConfig(
        {"start_date": "2024-01-05", "end_date": "2024-01-12", "max_staleness_calendar_days": 5},
        {},
        (source,),
    )
    path = tmp_path / "equity.parquet"
    pd.DataFrame({"date": ["2024-01-12"], "value": [100.0]}).to_parquet(path)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "series": [
                    {
                        "series_name": "equity",
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "unit": "index_points",
                    }
                ]
            }
        )
    )
    report = tmp_path / "report.json"
    with pytest.raises(DataQualityError):
        enforce_raw_controls(config, tmp_path, report)
    assert "series_identifier" in {x["rule"] for x in json.loads(report.read_text())["findings"]}
