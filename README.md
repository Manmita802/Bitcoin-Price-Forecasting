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
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -e ".[arima,xgboost,lstm]"
.\.venv\Scripts\python scripts\run_experiment.py
```

The runner reads `data/bitcoin_features.csv` and `configs/default.yaml` by default. Use `--data`, `--config`, and `--output-root` to override those paths. Relative paths resolve from the repository root.

The row dated *t* uses information available after candle *t* closes and predicts the close at *t+1*. Splits are based on target date: train through 2023-12-31, select settings in 2024, and reserve 2025 for the final evaluation. Test predictions are rolling one-step forecasts with fixed learned parameters; ARIMA updates its state after each observed close. The LSTM uses 30-day feature sequences and predicts log return, then converts the result to a close-price forecast. Every model is evaluated on the same test dates against the persistence baseline.

Run artifacts are saved under `artifacts/runs/` and excluded from Git. The data exports currently contain engineered features and targets; they are not immutable raw OHLCV snapshots.

## Run the API

Install the API dependencies and start FastAPI from the repository root:

```powershell
.\.venv\Scripts\python -m pip install -e ".[api,arima,xgboost,lstm]"
.\.venv\Scripts\python -m uvicorn btc_forecasting.api.main:app --reload --app-dir src
```

Open `http://127.0.0.1:8000/docs` for the interactive API description. Run the experiment to create deployable XGBoost and LSTM bundles. The API loads the latest run from `artifacts/runs/`; set `BTC_FORECASTING_MODEL_RUN` to select a particular run. The feature table path defaults to `data/bitcoin_features.csv` and can be changed with `BTC_FORECASTING_FEATURES`. For ML forecasts, the feature table and OHLCV history must end on the same date. XGBoost and LSTM currently support one-day-ahead forecasts only.

## Project limitations and next work

- The API does not support multi-day recursive XGBoost/LSTM inference because future OHLCV features are unknown.
- The API uses the historical ARIMA order file and has no authenticated access, rate limiting, or live data refresh.
- The checked-in feature CSVs duplicate processed data; preserving raw OHLCV and defining one canonical processed output remain follow-up tasks.
- Add model artifact versioning and serving after reviewing the saved experiment results. Then build the frontend and deployment workflow.

Keep the original preprocessing and ARIMA notebooks for exploration and historical reference. Use the shared package and script for repeatable experiments.
