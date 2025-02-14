import xgboost as xgb
import pandas as pd
from sklearn.model_selection import GridSearchCV
from .base_model import BaseModel

class XGBoostModel(BaseModel):
    """Modelo de forecasting con XGBoost usando lag features."""

    def __init__(self, lags=3, rolling_window=3):
        self.lags = lags
        self.rolling_window = rolling_window
        self.model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100)

    def get_param_grid(self):
        """Devuelve el espacio de búsqueda de hiperparámetros para XGBoost."""
        return {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.2]
        }

    def load_data(self, data, train_data=None):
        """Carga y preprocesa los datos para XGBoost, asegurando que los lags y rolling windows se generen correctamente.

        - `data`: Puede ser un archivo CSV o un DataFrame con los datos a procesar.
        - `train_data`: Opcional. Si se está preparando un conjunto de prueba (futuro), usar datos de entrenamiento para generar lags correctamente.
        """
        df = super().load_data(data)

        # Si estamos generando datos futuros, necesitamos referencia de train_data
        if train_data is not None:
            df_train = super().load_data(train_data)
            df = pd.concat([df_train, df]).reset_index(drop=True)

        # Generar lags como nuevas columnas
        for lag in range(1, self.lags + 1):
            df[f"Sales_Lag_{lag}"] = df["Sales"].shift(lag)

        # Agregar media móvil de las ventas como feature
        df["Sales_Rolling_Mean"] = df["Sales"].shift(1).rolling(self.rolling_window).mean()

        # Si se usó train_data, eliminamos las filas de entrenamiento y solo dejamos las nuevas predicciones
        if train_data is not None:
            df = df.iloc[-len(data):].reset_index(drop=True)

        # Para ser consistentes con los demas modelos
        df = df.rename(columns={"Sales": "y"})

        return df.dropna().reset_index(drop=True)

        

    def train(self, X, y):
        """Entrena XGBoost con datos estructurados."""
        self.model.fit(X, y)

    def predict(self, X):
        """Predice valores futuros usando el modelo entrenado."""
        return self.model.predict(X)

    def fine_tune(self, X, y, param_grid):
        """Ejecuta GridSearchCV para encontrar los mejores hiperparámetros."""
        grid_search = GridSearchCV(self.model, param_grid, cv=3, scoring="neg_mean_absolute_error", verbose=1)
        grid_search.fit(X, y)

        # Asignamos los mejores hiperparámetros encontrados al modelo
        self.model = grid_search.best_estimator_

        return grid_search.best_params_