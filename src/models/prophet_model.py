from prophet import Prophet
import pandas as pd
from itertools import product
from .base_model import BaseModel
from sklearn.metrics import mean_absolute_error

class ProphetModel(BaseModel):
    """Modelo de forecasting con Prophet."""

    def __init__(self):
        self.model = None  #Prophet() Model is instantiated in train()

    def get_param_grid(self):
        """Devuelve el espacio de búsqueda de hiperparámetros para Prophet."""
        return {
            "changepoint_prior_scale": [0.001, 0.01, 0.1, 0.5],
            "seasonality_mode": ["additive", "multiplicative"]
        }

    def load_data(self, data, train_data=None):
        """Carga y preprocesa datos para Prophet. Asegura que los datos futuros tengan fechas correctas."""
        df = super().load_data(data)

        if train_data is not None:
            train_data["ds"] = pd.date_range(start="2023-01-01", periods=len(train_data), freq="W")
            df["ds"] = pd.date_range(start=train_data["ds"].max() + pd.Timedelta(weeks=1), periods=len(df), freq="W")
        else:
            df["ds"] = pd.date_range(start="2023-01-01", periods=len(df), freq="W")

        df = df.rename(columns={"Sales": "y"})
        return df

    def train(self, X, y, changepoint_prior_scale=0.05, seasonality_mode="additive"):
        """Trains Prophet with given hyperparameters."""
        self.model = Prophet(changepoint_prior_scale=changepoint_prior_scale, seasonality_mode=seasonality_mode)
        df = X.copy()
        df["y"] = y
        self.model.fit(df)

    def predict(self, X):
        future = X.copy()  # Copiar para evitar modificar X directamente
        forecast = self.model.predict(future)
        return forecast["yhat"]  # Retornamos la predicción
    
    def fine_tune(self, X, y, param_grid):
        """Manually tunes Prophet hyperparameters."""
        best_score = float("inf")
        best_params = {}

        # Iterate over all combinations of hyperparameters
        for params in product(param_grid["changepoint_prior_scale"], param_grid["seasonality_mode"]):
            changepoint_prior, seasonality = params
            self.train(X, y, changepoint_prior_scale=changepoint_prior, seasonality_mode=seasonality)
            predictions = self.predict(X)
            mae = mean_absolute_error(y, predictions)

            if mae < best_score:
                best_score = mae
                best_params = {"changepoint_prior_scale": changepoint_prior, "seasonality_mode": seasonality}

        # Train the model with the best parameters
        self.train(X, y, **best_params)

        return best_params