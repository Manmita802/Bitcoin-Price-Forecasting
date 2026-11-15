"""Application services for health, model readiness, and forecasts."""

from __future__ import annotations

import importlib.util
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import load_ohlcv_csv
from ..models.arima import forecast_arima
from ..models.lstm import LSTMForecaster
from ..models.xgboost_model import XGBoostForecaster
from .schemas import (
    ForecastPoint,
    ForecastRequest,
    ForecastResponse,
    ModelStatus,
    ReadinessResponse,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "bitcoin_data.csv"
DEFAULT_FEATURES_PATH = PROJECT_ROOT / "data" / "bitcoin_features.csv"
DEFAULT_RUNS_PATH = PROJECT_ROOT / "artifacts" / "runs"
ARIMA_ORDER_PATH = PROJECT_ROOT / "arima" / "arima_metrics.json"
DATA_ENV_VAR = "BTC_FORECASTING_DATA"
FEATURES_ENV_VAR = "BTC_FORECASTING_FEATURES"
MODEL_RUN_ENV_VAR = "BTC_FORECASTING_MODEL_RUN"
_MODEL_CACHE: dict[tuple[str, int], object] = {}


class DataUnavailableError(RuntimeError):
    """The configured historical data cannot support a forecast."""


class ModelUnavailableError(RuntimeError):
    """The requested model is not implemented or its dependency is missing."""


class ForecastUnavailableError(RuntimeError):
    """An implemented model cannot produce a forecast for the current input."""


def data_path() -> Path:
    """Return the configured dataset path or the repository's current CSV.

    Relative paths in ``BTC_FORECASTING_DATA`` are anchored at the project
    root so the selected dataset does not depend on the server's working dir.
    """
    configured = os.environ.get(DATA_ENV_VAR)
    if not configured:
        return DEFAULT_DATA_PATH

    path = Path(configured).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def features_path() -> Path:
    """Return the processed feature table used by trained ML models."""
    configured = os.environ.get(FEATURES_ENV_VAR)
    if not configured:
        return DEFAULT_FEATURES_PATH
    path = Path(configured).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _model_run_path(model_name: str) -> Path | None:
    configured = os.environ.get(MODEL_RUN_ENV_VAR)
    if configured:
        path = Path(configured).expanduser()
        return path if path.is_absolute() else PROJECT_ROOT / path
    if not DEFAULT_RUNS_PATH.is_dir():
        return None
    candidates = sorted(
        (path for path in DEFAULT_RUNS_PATH.iterdir() if path.is_dir()),
        key=lambda path: path.name,
        reverse=True,
    )
    for candidate in candidates:
        try:
            metadata = json.loads((candidate / "metadata.json").read_text(encoding="utf-8"))
            relative_path = metadata["model_artifacts"][model_name]
            model_path = (candidate / relative_path).resolve()
            model_path.relative_to(candidate.resolve())
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
        if model_path.is_file():
            return candidate
    return None


def _model_bundle(name: str) -> tuple[Path, dict]:
    run_path = _model_run_path(name)
    if run_path is None or not run_path.is_dir():
        raise ModelUnavailableError(
            f"No experiment run is available. Run the experiment and set {MODEL_RUN_ENV_VAR} if needed."
        )
    try:
        metadata = json.loads((run_path / "metadata.json").read_text(encoding="utf-8"))
        relative_path = metadata["model_artifacts"][name]
        model_path = (run_path / relative_path).resolve()
        model_path.relative_to(run_path.resolve())
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ModelUnavailableError(
            f"The selected experiment run has no valid {name} model bundle."
        ) from exc
    if not model_path.is_file():
        raise ModelUnavailableError(f"The saved {name} model bundle is missing: {model_path.name}")
    return model_path, metadata


def _load_feature_history(
    path: Path,
    *,
    feature_columns: list[str],
    expected_last_date: pd.Timestamp,
) -> pd.DataFrame:
    """Load the processed table and enforce its model-input schema and origin."""
    if not path.is_file():
        raise DataUnavailableError(
            f"Processed model features are unavailable at {path}. Set {FEATURES_ENV_VAR}."
        )
    try:
        frame = pd.read_csv(path)
        if "Date" not in frame.columns:
            raise ValueError("feature CSV must contain a Date column")
        dates = pd.to_datetime(frame.pop("Date"), errors="raise", utc=True)
        frame.index = dates.dt.tz_convert(None).dt.normalize()
        frame.index.name = "Date"
    except (OSError, ValueError, KeyError) as exc:
        raise DataUnavailableError(f"Model feature table could not be loaded: {exc}") from exc

    if frame.empty or frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise DataUnavailableError("Model feature dates must be non-empty, unique, and chronological.")
    if not frame.index[-1] == expected_last_date:
        raise DataUnavailableError(
            "OHLCV and processed feature data must end on the same date before forecasting."
        )
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise DataUnavailableError(f"Model feature table is missing columns: {', '.join(missing)}")

    try:
        values = frame.loc[:, feature_columns].apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise DataUnavailableError(f"Model features must be numeric: {exc}") from exc
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise DataUnavailableError("Latest model features contain missing or non-finite values.")
    return values


def _cached_model(name: str, path: Path, metadata: dict):
    cache_key = (str(path), path.stat().st_mtime_ns)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]
    try:
        if name == "xgboost":
            settings = metadata["model"]["xgboost"]
            model = XGBoostForecaster.load(
                path,
                feature_columns=metadata["feature_columns"],
                parameters=settings["parameters"],
            )
        else:
            model = LSTMForecaster.load(path)
    except (ImportError, OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise ModelUnavailableError(f"The saved {name} model could not be loaded: {exc}") from exc
    _MODEL_CACHE.clear()
    _MODEL_CACHE[cache_key] = model
    return model


def list_models() -> list[ModelStatus]:
    """Report implemented models separately from optional runtime readiness."""
    arima_ready = False
    if importlib.util.find_spec("statsmodels") is None:
        arima_detail = "Install the project's arima dependency group to enable ARIMA."
    else:
        try:
            _saved_arima_order()
        except ModelUnavailableError as exc:
            arima_detail = str(exc)
        else:
            arima_ready = True
            arima_detail = "Fits the saved ARIMA order to the available close history."

    ml_status: dict[str, tuple[bool, str]] = {}
    for name, dependencies in {
        "xgboost": ("xgboost", "sklearn"),
        "lstm": ("torch", "joblib", "sklearn"),
    }.items():
        missing_dependencies = [
            dependency
            for dependency in dependencies
            if importlib.util.find_spec(dependency) is None
        ]
        if missing_dependencies:
            ml_status[name] = (
                False,
                f"Install the project's {name} dependency group to enable serving.",
            )
            continue
        try:
            _model_bundle(name)
        except ModelUnavailableError as exc:
            ml_status[name] = (False, str(exc))
        else:
            ml_status[name] = (
                True,
                "Loads the selected saved model bundle and serves one-day-ahead forecasts.",
            )

    return [
        ModelStatus(
            name="persistence",
            implemented=True,
            ready=True,
            detail="Repeats the latest observed close; available as a baseline.",
        ),
        ModelStatus(
            name="arima",
            implemented=True,
            ready=arima_ready,
            detail=arima_detail,
        ),
        ModelStatus(
            name="xgboost",
            implemented=True,
            ready=ml_status["xgboost"][0],
            detail=ml_status["xgboost"][1],
        ),
        ModelStatus(
            name="lstm",
            implemented=True,
            ready=ml_status["lstm"][0],
            detail=ml_status["lstm"][1],
        ),
    ]


def _saved_arima_order() -> tuple[int, int, int]:
    try:
        metrics = json.loads(ARIMA_ORDER_PATH.read_text(encoding="utf-8-sig"))
        raw_order = metrics["model_order"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ModelUnavailableError(
            "The saved ARIMA model order is missing or invalid."
        ) from exc

    if (
        not isinstance(raw_order, list)
        or len(raw_order) != 3
        or any(type(value) is not int or value < 0 for value in raw_order)
    ):
        raise ModelUnavailableError("The saved ARIMA model order must contain three non-negative integers.")
    return tuple(raw_order)  # type: ignore[return-value]


def _load_history() -> tuple[Path, pd.DataFrame]:
    """Load and validate the configured daily OHLCV history."""
    path = data_path()
    if not path.is_file():
        raise DataUnavailableError(
            f"Historical BTC data is unavailable. Set {DATA_ENV_VAR} to a CSV path."
        )
    try:
        history = load_ohlcv_csv(path)
    except (OSError, ValueError, KeyError) as exc:
        raise DataUnavailableError(f"Historical BTC data could not be loaded: {exc}") from exc
    return path, history


def check_readiness() -> ReadinessResponse:
    """Confirm that a valid daily dataset is available to forecast from."""
    path, history = _load_history()
    return ReadinessResponse(
        status="ready",
        data_source=path.name,
        history_start=history.index[0].date(),
        history_end=history.index[-1].date(),
    )


def create_forecast(request: ForecastRequest) -> ForecastResponse:
    """Build a response from the local historical snapshot and chosen model."""
    path, history = _load_history()

    close = history["Close"].astype(float)
    if close.empty:
        raise DataUnavailableError("Historical BTC data has no close prices.")

    lower = upper = None
    if request.model == "persistence":
        predicted_values = np.repeat(float(close.iloc[-1]), request.horizon_days)
    elif request.model == "arima":
        if importlib.util.find_spec("statsmodels") is None:
            raise ModelUnavailableError("Install the project's arima dependency group to enable ARIMA.")
        order = _saved_arima_order()
        try:
            predicted, lower, upper = forecast_arima(close, request.horizon_days, order=order)
        except (ValueError, RuntimeError, np.linalg.LinAlgError, FloatingPointError) as exc:
            raise ForecastUnavailableError(f"ARIMA forecast failed: {exc}") from exc
        predicted_values = predicted.to_numpy(dtype=float)
    elif request.model in {"xgboost", "lstm"}:
        if request.horizon_days != 1:
            raise ModelUnavailableError(
                f"{request.model} currently supports one-day-ahead forecasts only; "
                "multi-day inference needs a defined future-feature strategy."
            )
        if request.model == "xgboost" and importlib.util.find_spec("xgboost") is None:
            raise ModelUnavailableError("Install the project's xgboost dependency group to enable XGBoost.")
        if request.model == "lstm" and any(
            importlib.util.find_spec(dependency) is None
            for dependency in ("torch", "joblib", "sklearn")
        ):
            raise ModelUnavailableError("Install the project's lstm dependency group to enable LSTM.")

        model_path, metadata = _model_bundle(request.model)
        model = _cached_model(request.model, model_path, metadata)
        feature_columns = metadata.get("feature_columns")
        if not isinstance(feature_columns, list) or not feature_columns:
            raise ModelUnavailableError("The saved run does not contain a valid model feature schema.")
        feature_history = _load_feature_history(
            features_path(),
            feature_columns=feature_columns,
            expected_last_date=history.index[-1],
        )

        try:
            if request.model == "xgboost":
                predicted = model.predict(feature_history.iloc[[-1]])
                predicted_values = predicted.to_numpy(dtype=float)
            else:
                sequence_length = int(metadata["model"]["lstm"]["sequence_length"])
                if len(feature_history) < sequence_length:
                    raise ValueError("Feature history is shorter than the saved LSTM sequence length")
                values = feature_history.tail(sequence_length).to_numpy(dtype=np.float32)[None, :, :]
                predicted_log_return = float(model.predict(values).iloc[0])
                predicted_values = np.asarray(
                    [float(close.iloc[-1]) * np.exp(predicted_log_return)],
                    dtype=float,
                )
        except (ValueError, RuntimeError, KeyError, OverflowError, FloatingPointError) as exc:
            raise ForecastUnavailableError(f"{request.model} forecast failed: {exc}") from exc
    else:
        raise ModelUnavailableError(f"The {request.model} model is not implemented for serving yet.")

    predicted_values = np.asarray(predicted_values, dtype=float)
    if predicted_values.ndim != 1 or len(predicted_values) != request.horizon_days:
        raise DataUnavailableError(
            "The selected model returned a forecast with an unexpected length."
        )
    if not np.isfinite(predicted_values).all():
        raise DataUnavailableError("The selected model returned a non-finite forecast.")

    lower_values = upper_values = None
    if lower is not None or upper is not None:
        if lower is None or upper is None:
            raise DataUnavailableError(
                "The selected model returned an incomplete confidence interval."
            )
        lower_values = np.asarray(lower, dtype=float)
        upper_values = np.asarray(upper, dtype=float)
        if (
            lower_values.ndim != 1
            or upper_values.ndim != 1
            or len(lower_values) != request.horizon_days
            or len(upper_values) != request.horizon_days
        ):
            raise DataUnavailableError(
                "The selected model returned confidence intervals with an unexpected length."
            )
        if not np.isfinite(lower_values).all() or not np.isfinite(upper_values).all():
            raise DataUnavailableError(
                "The selected model returned non-finite confidence intervals."
            )
        if np.any(lower_values > upper_values):
            raise DataUnavailableError(
                "The selected model returned invalid confidence interval bounds."
            )

    start = history.index[0]
    last_date = history.index[-1]
    future_dates = [last_date.date() + timedelta(days=step) for step in range(1, request.horizon_days + 1)]
    points = [
        ForecastPoint(
            date=forecast_date,
            predicted_close_usd=float(predicted_values[index]),
            lower_95_usd=float(lower_values[index]) if lower_values is not None else None,
            upper_95_usd=float(upper_values[index]) if upper_values is not None else None,
        )
        for index, forecast_date in enumerate(future_dates)
    ]
    return ForecastResponse(
        symbol=request.symbol,
        model=request.model,
        horizon_days=request.horizon_days,
        generated_at_utc=datetime.now(timezone.utc),
        history_start=start.date(),
        history_end=last_date.date(),
        last_observed_close_usd=float(close.iloc[-1]),
        data_source=path.name,
        forecasts=points,
    )
