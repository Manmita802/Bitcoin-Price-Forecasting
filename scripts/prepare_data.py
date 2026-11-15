"""Download or load OHLCV, preserve a raw snapshot, and write model features."""

from __future__ import annotations

import argparse
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from btc_forecasting.data import fetch_ohlcv, load_ohlcv_csv, save_raw_snapshot
from btc_forecasting.features import make_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or not isinstance(config.get("data"), dict):
        raise ValueError("Config must contain a data mapping")
    return config


def prepare_data(
    config_path: Path,
    *,
    source_csv: Path | None = None,
    raw_output: Path | None = None,
    features_output: Path | None = None,
    overwrite_features: bool = False,
) -> tuple[Path, Path]:
    """Create one immutable OHLCV snapshot and one canonical feature CSV."""
    config = _load_config(config_path)
    settings = config["data"]
    try:
        symbol = str(settings["symbol"])
        start = str(settings["start_date"])
        end_exclusive = str(settings["end_date_exclusive"])
        raw_directory = _project_path(settings["raw_dir"])
        features_path = features_output or _project_path(settings["features_file"])
    except KeyError as exc:
        raise ValueError(f"Data config is missing {exc.args[0]!r}") from exc

    if features_path.exists() and not overwrite_features:
        raise FileExistsError(
            f"Processed feature file already exists: {features_path}. "
            "Choose another --features-output path or pass --overwrite-features."
        )

    if source_csv is None:
        raw = fetch_ohlcv(symbol=symbol, start=start, end_exclusive=end_exclusive)
        source_description = "yfinance"
    else:
        raw = load_ohlcv_csv(source_csv)
        source_description = str(source_csv)

    retrieved_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_path = raw_output or (
        raw_directory / f"{symbol}_{raw.index[0]:%Y%m%d}_{raw.index[-1]:%Y%m%d}_{retrieved_at}.csv"
    )
    raw_path = raw_path if raw_path.is_absolute() else _project_path(raw_path)
    if source_csv is not None and source_csv.resolve() == raw_path.resolve():
        snapshot_path = source_csv
    else:
        snapshot_path = save_raw_snapshot(raw, raw_path)

    features = make_features(raw, drop_incomplete=False)
    feature_columns = [column for column in features.columns if not column.startswith("Target_")]
    features = features.dropna(subset=feature_columns).copy()
    if features.empty:
        raise ValueError("Feature generation produced no complete feature rows")

    features_path.parent.mkdir(parents=True, exist_ok=True)
    if features_path.exists() and not overwrite_features:
        raise FileExistsError(
            f"Processed feature file already exists: {features_path}. "
            "Choose another --features-output path or pass --overwrite-features."
        )

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            suffix=".csv",
            prefix=".features-",
            dir=features_path.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            features.to_csv(stream, date_format="%Y-%m-%d", index_label="Date")
        if features_path.exists() and not overwrite_features:
            raise FileExistsError(f"Processed feature file already exists: {features_path}")
        temporary_path.replace(features_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    print(f"Source: {source_description}")
    print(f"Raw snapshot: {snapshot_path}")
    print(f"Features: {features_path}")
    print(f"Rows: {len(features):,}; dates: {features.index[0]:%Y-%m-%d} to {features.index[-1]:%Y-%m-%d}")
    return snapshot_path, features_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-csv", type=Path, help="Use a local OHLCV CSV instead of yfinance")
    parser.add_argument("--raw-output", type=Path, help="Raw snapshot path; the file is never overwritten")
    parser.add_argument("--features-output", type=Path, help="Override the configured processed feature path")
    parser.add_argument(
        "--overwrite-features",
        action="store_true",
        help="Replace the selected processed feature file after generating a temporary copy",
    )
    args = parser.parse_args()
    prepare_data(
        _project_path(args.config),
        source_csv=_project_path(args.source_csv) if args.source_csv is not None else None,
        raw_output=_project_path(args.raw_output) if args.raw_output is not None else None,
        features_output=_project_path(args.features_output) if args.features_output is not None else None,
        overwrite_features=args.overwrite_features,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
