"""Small command-line interface for source auditing and raw downloads."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from eurostress.audit import generate_coverage_report
from eurostress.benchmarks import build_benchmark_dataset
from eurostress.config import load_config
from eurostress.db import initialize_database
from eurostress.evaluation import evaluate_locked_test, freeze_evaluation
from eurostress.exploratory import build_exploratory_models
from eurostress.features import build_feature_dataset
from eurostress.models import build_models
from eurostress.risk_validation import build_risk_validation
from eurostress.sources import EcbClient, YahooClient


def audit(config_path: str) -> int:
    config = load_config(config_path)
    print(f"Project: {config.project['name']}")
    print(f"Frozen sample: {config.project['start_date']} to {config.project['end_date']}")
    for series in config.series:
        identifier = series.key or series.symbol
        print(
            f"- {series.name}: provider={series.provider}, id={identifier}, "
            f"role={series.role}, redistribution={series.redistribution}"
        )
    return 0


def download(config_path: str, output_directory: str) -> int:
    config = load_config(config_path)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)

    start = config.project["start_date"]
    end = config.project["end_date"]
    ecb = EcbClient()
    yahoo = YahooClient()
    manifest_entries: list[dict[str, object]] = []

    for series in config.series:
        if series.provider == "ecb":
            frame = ecb.fetch(series.flow or "", series.key or "", start, end)
        elif series.provider == "yahoo":
            frame = yahoo.fetch(series.symbol or "", start, end)
        else:
            raise ValueError(f"Unsupported provider: {series.provider}")

        source_row_count = len(frame)
        dropped_missing_rows = int(frame["value"].isna().sum())
        if dropped_missing_rows:
            frame = frame.dropna(subset=["value"]).reset_index(drop=True)
        if frame.empty:
            raise ValueError(f"No usable observations remain for {series.name}")

        destination = output / f"{series.name}.parquet"
        frame.to_parquet(destination, index=False)
        source_reference = (
            ecb.build_url(series.flow or "", series.key or "", start, end)
            if series.provider == "ecb"
            else series.symbol
        )
        manifest_entries.append(
            {
                "series_name": series.name,
                "series_id": str(frame["series_id"].iloc[0]),
                "provider": series.provider,
                "source_reference": source_reference,
                "requested_start": start,
                "requested_end_inclusive": end,
                "observation_start": frame["date"].min().date().isoformat(),
                "observation_end": frame["date"].max().date().isoformat(),
                "row_count": len(frame),
                "source_row_count": source_row_count,
                "dropped_missing_rows": dropped_missing_rows,
                "path": str(destination),
                "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            }
        )
        print(
            f"Wrote {len(frame):,} rows to {destination} "
            f"({dropped_missing_rows} source rows with missing values excluded)"
        )

    manifest = {
        "project": config.project["name"],
        "created_at_utc": datetime.now(UTC).isoformat(),
        "series": manifest_entries,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote manifest to {manifest_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eurostress")
    parser.add_argument("--config", default="config/series.yml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("audit", help="Validate and display the configured sources")

    download_parser = subparsers.add_parser("download", help="Download raw source data")
    download_parser.add_argument("--output", default="data/raw")

    coverage_parser = subparsers.add_parser(
        "coverage", help="Audit weekly coverage without inspecting market outcomes"
    )
    coverage_parser.add_argument("--input", default="data/raw")
    coverage_parser.add_argument("--output", default="docs/raw_coverage_report.md")

    database_parser = subparsers.add_parser("init-db", help="Create the DuckDB schema")
    database_parser.add_argument("--database", default="data/eurostress.duckdb")

    features_parser = subparsers.add_parser(
        "build-features", help="Build the frozen point-in-time weekly feature table"
    )
    features_parser.add_argument("--input", default="data/raw")
    features_parser.add_argument("--output", default="data/processed")
    features_parser.add_argument("--database", default="data/eurostress.duckdb")

    benchmarks_parser = subparsers.add_parser(
        "build-benchmarks", help="Build next-week targets and point-in-time benchmarks"
    )
    benchmarks_parser.add_argument("--raw", default="data/raw")
    benchmarks_parser.add_argument("--processed", default="data/processed")
    benchmarks_parser.add_argument("--database", default="data/eurostress.duckdb")

    models_parser = subparsers.add_parser(
        "fit-models", help="Fit and select Phase 4 models using training/validation only"
    )
    models_parser.add_argument("--processed", default="data/processed")
    models_parser.add_argument("--models", default="data/models")
    models_parser.add_argument("--reports", default="reports")

    exploratory_parser = subparsers.add_parser(
        "fit-exploratory", help="Run labelled Phase 4B stabilization without locked-test access"
    )
    exploratory_parser.add_argument("--exploratory-config", default="config/exploratory.yml")
    exploratory_parser.add_argument("--processed", default="data/processed")
    exploratory_parser.add_argument("--output", default="data/exploratory")
    exploratory_parser.add_argument("--reports", default="reports")

    risk_parser = subparsers.add_parser(
        "risk-validation",
        help="Run the post-hoc VaR, ES and regime-conditional validation extension",
    )
    risk_parser.add_argument("--risk-config", default="config/risk_validation.yml")
    risk_parser.add_argument("--equity", default="data/raw/euro_stoxx_50.parquet")
    risk_parser.add_argument(
        "--regimes", default="data/evaluation/locked_forecasts_and_outcomes.parquet"
    )
    risk_parser.add_argument("--output", default="data/risk_validation")
    risk_parser.add_argument("--reports", default="reports")

    freeze_parser = subparsers.add_parser(
        "freeze-evaluation", help="Freeze final evaluation hashes without reading test rows"
    )
    freeze_parser.add_argument("--evaluation-config", default="config/evaluation_lock.yml")
    freeze_parser.add_argument("--output", default="protocol/final_evaluation_lock.json")

    evaluate_parser = subparsers.add_parser(
        "evaluate-locked", help="Run the frozen one-time locked-test evaluation"
    )
    evaluate_parser.add_argument("--evaluation-config", default="config/evaluation_lock.yml")
    evaluate_parser.add_argument("--lock", default="protocol/final_evaluation_lock.json")
    evaluate_parser.add_argument("--output", default="data/evaluation")
    unlock_group = evaluate_parser.add_mutually_exclusive_group()
    unlock_group.add_argument("--unlock-locked-test", action="store_true")
    unlock_group.add_argument("--reproduce-frozen-evaluation", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "audit":
        return audit(args.config)
    if args.command == "download":
        return download(args.config, args.output)
    if args.command == "coverage":
        config = load_config(args.config)
        results = generate_coverage_report(config, args.input, args.output)
        for result in results:
            print(f"{result.series_name}: {result.weekly_coverage_percent:.2f}% weekly coverage")
        print(f"Wrote coverage report to {args.output}")
        return 0
    if args.command == "init-db":
        path = initialize_database(args.database)
        print(f"Initialized {path}")
        return 0
    if args.command == "build-features":
        config = load_config(args.config)
        result = build_feature_dataset(
            config=config,
            config_path=args.config,
            input_directory=args.input,
            output_directory=args.output,
            database_path=args.database,
        )
        print(
            f"Built {result.rows:,} weekly rows from {result.first_information_date} "
            f"to {result.last_information_date} (run {result.run_id})"
        )
        print(
            f"Split counts: training={result.training_rows}, "
            f"validation={result.validation_rows}, locked_test={result.locked_test_rows}"
        )
        return 0
    if args.command == "build-benchmarks":
        config = load_config(args.config)
        result = build_benchmark_dataset(
            config=config,
            config_path=args.config,
            raw_directory=args.raw,
            processed_directory=args.processed,
            database_path=args.database,
        )
        print(
            f"Built {result.rows:,} structurally valid target/forecast rows from "
            f"{result.first_information_date} to {result.last_information_date} "
            f"(run {result.run_id})"
        )
        print(
            f"Split counts: training={result.training_rows}, "
            f"validation={result.validation_rows}, locked_test={result.locked_test_rows}"
        )
        print("Performance metrics computed: no")
        return 0
    if args.command == "fit-models":
        config = load_config(args.config)
        result = build_models(
            config=config,
            config_path=args.config,
            processed_directory=args.processed,
            model_directory=args.models,
            report_directory=args.reports,
        )
        print(
            f"Evaluated {result.candidate_count} model-family/state-count candidates; "
            f"{result.admissible_candidate_count} passed the frozen rules (run {result.run_id})"
        )
        for family, selection in result.selected_models.items():
            print(f"- {family}: {selection or 'no admissible candidate'}")
        print("Locked-test rows read: 0; target-performance metrics computed: no")
        return 0
    if args.command == "fit-exploratory":
        config = load_config(args.config)
        result = build_exploratory_models(
            core_config=config,
            exploratory_config_path=args.exploratory_config,
            processed_directory=args.processed,
            output_directory=args.output,
            report_directory=args.reports,
        )
        print(f"Completed exploratory stabilization run {result.run_id}")
        print(f"- PCA HMM: {result.pca_selected or 'no admissible candidate'}")
        print(f"- seed ensemble HMM: {result.ensemble_selected or 'no operational candidate'}")
        print("Classification: exploratory; locked-test rows read: 0; target metrics: no")
        return 0
    if args.command == "risk-validation":
        result = build_risk_validation(
            config_path=args.risk_config,
            equity_path=args.equity,
            regime_path=args.regimes,
            output_directory=args.output,
            report_directory=args.reports,
        )
        print(
            f"Validated {result.model_count} models across {result.evaluation_days:,} daily "
            f"returns from {result.evaluation_start} to {result.evaluation_end}"
        )
        print(f"Higher-risk regime days: {result.high_risk_days:,}")
        print(f"Report: {result.report_path}")
        print("Classification: post-hoc market-risk extension; original sealed result unchanged")
        return 0
    if args.command == "freeze-evaluation":
        destination = freeze_evaluation(args.evaluation_config, args.output)
        print(f"Frozen final evaluation at {destination}")
        print("Locked-test rows read: 0; target-performance metrics computed: no")
        return 0
    if args.command == "evaluate-locked":
        report = evaluate_locked_test(
            evaluation_config_path=args.evaluation_config,
            lock_path=args.lock,
            output_directory=args.output,
            unlock_locked_test=args.unlock_locked_test,
            reproduce_frozen_evaluation=args.reproduce_frozen_evaluation,
        )
        print(f"Wrote frozen evaluation to {report}")
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
