# Experiment runner

Use Python `>=3.11,<3.13`, preferably Python 3.12. From the repository root:

```powershell
python -m pip install -e ".[arima,xgboost,lstm]"
python scripts/run_experiment.py
```

The default input is `data/bitcoin_features.csv`; settings are read from `configs/default.yaml`. Use `--data`, `--config`, or `--output-root` to override them. Relative paths are resolved from the repository root.

The runner partitions rows by **target date**: train through 2023-12-31, use 2024 for setting selection, and reserve 2025 for final evaluation. All forecasts predict the next close after the current daily candle closes and are scored on the same target dates against persistence.

- ARIMA order is selected by AIC using training-period closes only. It uses rolling one-step forecasts and appends each observed close without refitting parameters.
- XGBoost estimator count is selected on validation. The selected model is refit on train plus validation before test prediction.
- LSTM epoch count is selected on validation. It uses causal 30-row sequences, predicts log return, and fits feature and target scalers only on its training rows during selection. The selected model is then refit on train plus validation.

Each output directory contains test predictions, shared test metrics, candidate validation scores, effective configuration, input SHA-256, and software versions. Set `--output-root` to choose an artifact location. The default `artifacts/runs/` output is excluded from Git.
