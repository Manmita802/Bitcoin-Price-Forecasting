# Experiment runner

Use Python `>=3.11,<3.13`, preferably Python 3.12. From the repository root:

```powershell
uv python install 3.12
uv sync --python 3.12 --extra arima --extra xgboost --extra lstm
uv run python scripts/prepare_data.py
uv run python scripts/run_experiment.py
```

`prepare_data.py` downloads the configured yfinance history, saves a timestamped immutable OHLCV snapshot under `data/raw/`, and writes the canonical feature table at `data/processed/bitcoin_features.csv`. Existing processed output is protected by default; use `--overwrite-features` to replace it deliberately. Use `--source-csv` to prepare from a local OHLCV file instead. The experiment reads the configured feature path and model settings from `configs/default.yaml`. Use `--data`, `--config`, or `--output-root` to override runner paths. Relative paths are resolved from the repository root.

The runner partitions rows by **target date**: train through 2023-12-31, use 2024 for setting selection, and reserve 2025 for final evaluation. All forecasts predict the next close after the current daily candle closes and are scored on the same target dates against persistence.

- ARIMA order is selected by AIC using training-period closes only. It uses rolling one-step forecasts and appends each observed close without refitting parameters.
- XGBoost estimator count is selected on validation. The selected model is refit on train plus validation before test prediction.
- LSTM epoch count is selected on validation. It uses causal 30-row sequences, predicts log return, and fits feature and target scalers only on its training rows during selection. The selected model is then refit on train plus validation.

Each output directory contains test predictions, shared test metrics, candidate validation scores, effective configuration, input SHA-256, and software versions. Set `--output-root` to choose an artifact location. The default `artifacts/runs/` output is excluded from Git.
