from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm
import pandas as pd
from .base_model import BaseModel

class SARIMAModel(BaseModel):
    """Implementación de SARIMA para forecasting."""

    def __init__(self, order=(1, 0, 1), seasonal_order=(0, 0, 0, 4)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None

    def get_param_grid(self):
        """SARIMA usa auto_arima, por lo que no necesita un grid de parámetros."""
        return None
    
    def load_data(self, data, train_data=None):
        """Carga y preprocesa datos para SARIMA. Se asegura de mantener la continuidad de fechas."""
        df = super().load_data(data)

        if train_data is not None:
            # Asegurar que train_data tenga un índice de fechas
            train_data = train_data.copy()
            train_data.index = pd.date_range(start="2023-01-01", periods=len(train_data), freq="W")

            # Asignar nuevas fechas para predicción
            df.index = pd.date_range(start=train_data.index.max() + pd.Timedelta(weeks=1), periods=len(df), freq="W")
        else:
            df.index = pd.date_range(start="2023-01-01", periods=len(df), freq="W")

        return df.rename(columns={"Sales": "y"})
    
    def train(self, X, y, order=None, seasonal_order=None):
        """Trains SARIMA with given hyperparameters."""
        if order is None:
            order = self.order
        if seasonal_order is None:
            seasonal_order = self.seasonal_order

        self.model = SARIMAX(y, order=order, seasonal_order=seasonal_order, enforce_stationarity=False).fit(method="nm", disp=False)


    def predict(self, X):
        return self.model.forecast(steps=len(X))
    
    def fine_tune(self, X, y, param_grid=None):
        """Tunes SARIMA using auto_arima."""
        self.model = pm.auto_arima(
            y,
            seasonal=True,
            stepwise=True,
            trace=True,
            error_action="ignore",  # Evita fallos si no encuentra buen modelo
            suppress_warnings=False,  # Silencia warnings de convergencia
            max_p=2, max_q=2, max_P=1, max_Q=1,  # Evita sobreajuste con pocos datos
            m=4,  # Estacionalidad semanal
            method="nm",  # Usar optimización más estable
        )

        # Get best parameters
        best_order = self.model.order
        best_seasonal_order = self.model.seasonal_order

        self.train(X, y, order=best_order, seasonal_order=best_seasonal_order)

        return {"order": best_order, "seasonal_order": best_seasonal_order}
