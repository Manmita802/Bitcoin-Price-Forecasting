# BTC-USD Price Forecasting

This repository contains a reproducible daily Bitcoin close forecasting backend. It keeps the existing notebooks and data exports while moving reusable forecasting work into Python modules.

## Current capabilities

- Validates and loads daily BTC-USD OHLCV data, builds causal technical and calendar features, and creates a next-day close target.
- Runs a chronological one-day-ahead comparison of persistence, ARIMA, XGBoost, and LSTM against the same 2025 test dates.
- Selects ARIMA order on training data only, and selects XGBoost estimator count and LSTM epoch count on 2024 validation data.
- Fits learned preprocessing on training data during model selection, then refits the selected model on train plus validation before final test inference.
- Saves per-date predictions, test metrics, validation candidate scores, configuration, input hash, and package versions in a unique run directory.
- Provides a local FastAPI backend for readiness, model status, persistence, ARIMA, and one-day XGBoost/LSTM forecasts when saved bundles are available.

## Environment and experiment

Use Python `>=3.11,<3.13` (Python 3.12 recommended). From the repository root:

```powershell
uv python install 3.12
uv sync --python 3.12 --extra api --extra arima --extra xgboost --extra lstm
uv run python scripts\prepare_data.py
uv run python scripts\run_experiment.py
```

`prepare_data.py` downloads the configured range from yfinance by default, stores a timestamped raw snapshot under `data/raw/`, and writes causal features to `data/processed/bitcoin_features.csv`. It preserves existing processed outputs unless you explicitly pass `--overwrite-features`. Use `--source-csv` to build features from an existing OHLCV CSV. The experiment runner then reads the configured feature path. Use `--data`, `--config`, and `--output-root` to override runner paths. Relative paths resolve from the repository root.

The row dated *t* uses information available after candle *t* closes and predicts the close at *t+1*. Splits are based on target date: train through 2023-12-31, select settings in 2024, and reserve 2025 for the final evaluation. Test predictions are rolling one-step forecasts with fixed learned parameters; ARIMA updates its state after each observed close. The LSTM uses 30-day feature sequences and predicts log return, then converts the result to a close-price forecast. Every model is evaluated on the same test dates against the persistence baseline.

Run artifacts are saved under `artifacts/runs/` and excluded from Git. The data exports currently contain engineered features and targets; they are not immutable raw OHLCV snapshots.

## Run the API

Install the API dependencies and start FastAPI from the repository root:

```powershell
uv run python -m uvicorn btc_forecasting.api.main:app --reload --app-dir src
```

Open `http://127.0.0.1:8000/docs` for the interactive API description. Run data preparation and the experiment to create the raw snapshot, feature table, and deployable XGBoost/LSTM bundles. The API automatically uses the newest timestamped raw snapshot under `data/raw/` and latest run under `artifacts/runs/`. Set `BTC_FORECASTING_MODEL_RUN` to select a particular run and `BTC_FORECASTING_DATA` to override the raw snapshot. The feature table defaults to `data/processed/bitcoin_features.csv`; set `BTC_FORECASTING_FEATURES` to override it. XGBoost and LSTM currently support one-day-ahead forecasts only.

## Project limitations and next work

- The API does not support multi-day recursive XGBoost/LSTM inference because future OHLCV features are unknown.
- The API uses the historical ARIMA order file and has no authenticated access, rate limiting, or live data refresh.
- The checked-in feature CSVs duplicate processed data; preserving raw OHLCV and defining one canonical processed output remain follow-up tasks.
- Add model artifact versioning and serving after reviewing the saved experiment results. Then build the frontend and deployment workflow.

Keep the original preprocessing and ARIMA notebooks for exploration and historical reference. Use the shared package and script for repeatable experiments.
