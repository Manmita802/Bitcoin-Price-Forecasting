"""Training-scaled LSTM adapter for supervised daily forecast sequences."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


class LSTMForecaster:
    """Fit a small LSTM regressor on fixed-length, already aligned sequences.

    Feature and target scalers are fit only from the arrays passed to ``fit``.
    Callers are responsible for constructing causal windows and choosing the
    training dates; ``predict`` applies the stored training scalers unchanged.
    """

    def __init__(
        self,
        *,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
        learning_rate: float = 0.001,
        batch_size: int = 64,
        epochs: int = 20,
        random_state: int = 42,
    ) -> None:
        if type(hidden_size) is not int or hidden_size < 1:
            raise ValueError("hidden_size must be a positive integer")
        if type(num_layers) is not int or num_layers < 1:
            raise ValueError("num_layers must be a positive integer")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if not 0 < learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1]")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if type(epochs) is not int or epochs < 1:
            raise ValueError("epochs must be a positive integer")

        self.parameters = {
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": epochs,
            "random_state": random_state,
        }
        self._model = None
        self._feature_scaler = None
        self._target_scaler = None
        self._sequence_shape: tuple[int, int] | None = None

    @staticmethod
    def _validate_sequences(sequences: np.ndarray | Sequence) -> np.ndarray:
        values = np.asarray(sequences, dtype=np.float32)
        if values.ndim != 3 or not values.shape[0] or not values.shape[1] or not values.shape[2]:
            raise ValueError("sequences must have shape (samples, time_steps, features)")
        if not np.isfinite(values).all():
            raise ValueError("sequences must contain only finite values")
        return values

    def fit(
        self,
        sequences: np.ndarray | Sequence,
        target: pd.Series | np.ndarray | Sequence[float],
    ) -> LSTMForecaster:
        """Train using aligned sequences and a one-dimensional regression target."""
        x = self._validate_sequences(sequences)
        y = np.asarray(target, dtype=np.float64).reshape(-1, 1)
        if len(x) != len(y) or not len(y):
            raise ValueError("sequences and target must have equal non-zero lengths")
        if not np.isfinite(y).all():
            raise ValueError("target must contain only finite values")

        try:
            import torch
            from sklearn.preprocessing import StandardScaler
        except ImportError as exc:
            raise ImportError("Install the project's lstm dependency group to use LSTM.") from exc

        torch.manual_seed(self.parameters["random_state"])
        self._feature_scaler = StandardScaler()
        scaled_x = self._feature_scaler.fit_transform(
            x.reshape(-1, x.shape[-1]).astype(np.float64)
        ).reshape(x.shape).astype(np.float32)
        self._target_scaler = StandardScaler()
        scaled_y = self._target_scaler.fit_transform(y).astype(np.float32)
        self._sequence_shape = (x.shape[1], x.shape[2])

        model_class = self._make_model(torch, input_size=x.shape[2])
        model = model_class()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.parameters["learning_rate"]
        )
        loss_function = torch.nn.MSELoss()
        tensor_x = torch.from_numpy(scaled_x)
        tensor_y = torch.from_numpy(scaled_y)
        generator = torch.Generator().manual_seed(self.parameters["random_state"])
        dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.parameters["batch_size"],
            shuffle=True,
            generator=generator,
        )

        model.train()
        for _ in range(self.parameters["epochs"]):
            for batch_x, batch_y in loader:
                optimizer.zero_grad(set_to_none=True)
                prediction = model(batch_x)
                loss = loss_function(prediction, batch_y)
                loss.backward()
                optimizer.step()

        model.eval()
        self._model = model
        return self

    def _make_model(self, torch, *, input_size: int):
        parameters = self.parameters

        class SequenceRegressor(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.recurrent = torch.nn.LSTM(
                    input_size=input_size,
                    hidden_size=parameters["hidden_size"],
                    num_layers=parameters["num_layers"],
                    batch_first=True,
                    dropout=(parameters["dropout"] if parameters["num_layers"] > 1 else 0.0),
                )
                self.output = torch.nn.Linear(parameters["hidden_size"], 1)

            def forward(self, values):
                recurrent_output, _ = self.recurrent(values)
                return self.output(recurrent_output[:, -1, :])

        return SequenceRegressor

    def predict(
        self,
        sequences: np.ndarray | Sequence,
        *,
        index: pd.Index | None = None,
    ) -> pd.Series:
        """Predict regression targets while preserving optional decision dates."""
        if (
            self._model is None
            or self._feature_scaler is None
            or self._target_scaler is None
            or self._sequence_shape is None
        ):
            raise RuntimeError("fit must be called before predict")

        x = self._validate_sequences(sequences)
        if x.shape[1:] != self._sequence_shape:
            raise ValueError(
                f"sequence shape differs from training; expected {self._sequence_shape}, "
                f"got {x.shape[1:]}"
            )
        if index is not None and len(index) != len(x):
            raise ValueError("prediction index length must match the number of sequences")

        import torch

        scaled_x = self._feature_scaler.transform(
            x.reshape(-1, x.shape[-1]).astype(np.float64)
        ).reshape(x.shape).astype(np.float32)
        self._model.eval()
        with torch.no_grad():
            scaled_prediction = self._model(torch.from_numpy(scaled_x)).cpu().numpy()
        prediction = self._target_scaler.inverse_transform(scaled_prediction).reshape(-1)
        if not np.isfinite(prediction).all():
            raise ValueError("LSTM returned non-finite predictions")
        prediction_index = index if index is not None else pd.RangeIndex(len(prediction))
        return pd.Series(prediction, index=prediction_index, name="prediction")
