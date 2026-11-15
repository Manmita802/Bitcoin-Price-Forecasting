# Forecasting API

This is a local development API for the BTC-USD project. It serves forecasts from the repository's current historical CSV; it does not download fresh market data. The configured default is `data/bitcoin_data.csv`. Set `BTC_FORECASTING_DATA` to use another CSV with a `Date` column and valid daily OHLCV columns. Relative values are resolved from the repository root; absolute paths are used as given.

## Install and run

Use a Python version allowed by the project (`>=3.11,<3.13`). From the repository root:

```powershell
uv python install 3.12
uv sync --python 3.12 --extra api --extra arima --extra xgboost --extra lstm
uv run python scripts/prepare_data.py
uv run python scripts/run_experiment.py
uv run python -m uvicorn btc_forecasting.api.main:app --reload --app-dir src
```

The interactive API description is available at `http://127.0.0.1:8000/docs` while the local server is running.

## Endpoints

- `GET /health` reports process health and whether the configured data file exists.
- `GET /ready` loads and validates the configured daily OHLCV data; it returns `503` if the API cannot serve forecasts from that dataset.
- `GET /api/v1/models` lists model implementation and dependency readiness.
- `POST /api/v1/forecast` accepts a BTC-USD model name and a one-to-90-day horizon.

Example request:

```json
{
  "symbol": "BTC-USD",
  "model": "persistence",
  "horizon_days": 7
}
```

`persistence` repeats the latest observed close and serves as a baseline. `arima` fits the order recorded in `arima/arima_metrics.json` to the local close history. Both use the last date in the CSV as the forecast origin.

`prepare_data.py` downloads the configured source range, saves a timestamped raw OHLCV snapshot in `data/raw/`, and writes `data/processed/bitcoin_features.csv`. It refuses to overwrite that feature file unless `--overwrite-features` is passed. The API automatically uses the newest raw snapshot and the configured feature table. Set `BTC_FORECASTING_DATA` or `BTC_FORECASTING_FEATURES` to choose paths manually.

XGBoost and LSTM require saved model bundles from a completed experiment run. The API selects the latest compatible directory under `artifacts/runs/` by default. Set `BTC_FORECASTING_MODEL_RUN` to a specific run directory when needed. The OHLCV and feature CSVs must end on the same date. These models currently serve one-day-ahead forecasts only because longer horizons need a defined future-feature strategy. The API returns a clear unavailable response when the bundle, dependency, or matching feature data is missing.

Create model bundles by running `python scripts/run_experiment.py`. Test metrics are computed before separate deployment bundles are fit on all labeled rows; those bundles are intended for inference and must not be used to reproduce the held-out 2025 metrics.

This API is a local development service. It has no authentication, rate limiting, CORS policy, live data refresh, or deployment configuration. Set those policies when the consuming application and hosting environment are defined.
