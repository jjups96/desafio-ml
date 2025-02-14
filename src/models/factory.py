from .xgb_model import XGBoostModel
from .prophet_model import ProphetModel
from .sarima_model import SARIMAModel
from .ridge_model import RidgeRegressionModel

class ModelFactory:
    """Fábrica para crear modelos dinámicamente."""
    
    _models = {
        "XGBoost": XGBoostModel,
        "Prophet": ProphetModel,
        "SARIMA": SARIMAModel,
        "Ridge": RidgeRegressionModel
    }

    @staticmethod
    def get_model(model_name):
        try:
            return ModelFactory._models[model_name]()
        except KeyError:
            raise ValueError(f"Modelo '{model_name}' no soportado")
