from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score, mean_absolute_error, root_mean_squared_error
import pandas as pd

class BaseModel(ABC):
    """Interfaz común para todos los modelos."""
    
    @abstractmethod
    def train(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def fine_tune(self, X, y, param_grid):
        pass

    def evaluate(self, X, y):
        predictions = self.predict(X)
        mae = mean_absolute_error(y, predictions)
        rmse = root_mean_squared_error(y, predictions)

        return mae, rmse

    def load_data(self, data):
        """Carga y preprocesa los datos. Puede recibir un archivo CSV o un DataFrame."""
        if isinstance(data, str):  # Si es una ruta, cargamos el CSV
            df = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):  # Si es un DataFrame, lo usamos directamente
            df = data.copy()
        else:
            raise ValueError("El argumento 'data' debe ser una ruta de archivo o un DataFrame.")

        return df
