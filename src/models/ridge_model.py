from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
import pandas as pd
from .base_model import BaseModel

class RidgeRegressionModel(BaseModel):
    """Modelo de forecasting usando Ridge Regression."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.model = Ridge(alpha=alpha)

    def get_param_grid(self):
        """Devuelve el espacio de búsqueda de hiperparámetros para XGBoost."""
        return {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}

    def load_data(self, data, train_data=None):
        """Carga y preprocesa los datos para Ridge Regression (usando lags)."""
        df = super().load_data(data)

        if train_data is not None:
            # Usar los últimos valores de entrenamiento para calcular los lags
            df_train = super().load_data(train_data)
            df = pd.concat([df_train, df]).reset_index(drop=True)

        # Generar lags (importante para forecasting)
        for lag in range(1, 5):  # Usamos 4 semanas de historial
            df[f"Sales_Lag_{lag}"] = df["Sales"].shift(lag)

        df = df.rename(columns={"Sales": "y"})

        # Si se usó train_data, eliminamos las filas de entrenamiento y solo dejamos las nuevas predicciones
        if train_data is not None:
            df = df.iloc[-len(data):].reset_index(drop=True)

        return df.dropna().reset_index(drop=True)

    def train(self, X, y):
        """Entrena Ridge Regression con los datos de entrada."""
        self.model.fit(X, y)

    def predict(self, X):
        """Predice las ventas futuras usando Ridge Regression."""
        return self.model.predict(X)
    
    def fine_tune(self, X, y, param_grid=None):
        """Usa GridSearchCV para optimizar el hiperparámetro alpha."""
        if param_grid is None:
            param_grid = {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}

        grid_search = GridSearchCV(Ridge(), param_grid, cv=3, scoring="neg_mean_absolute_error", verbose=1)
        grid_search.fit(X, y)

        # Asignamos el mejor hiperparámetro encontrado
        self.alpha = grid_search.best_params_["alpha"]
        self.model = grid_search.best_estimator_

        return {"alpha": self.alpha}
