import logging
from sklearn.model_selection import GridSearchCV

class Trainer:
    """Entrena modelos con logging."""

    def __init__(self, model):
        """
        Inicializa el entrenador con el modelo especificado.

        Parameters:
        model: Modelo a entrenar.
        """
        self.model = model
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def train(self, X, y):
        """
        Entrena el modelo con los datos proporcionados.

        Parameters:
        X (array-like): Datos de entrada para el entrenamiento.
        y (array-like): Etiquetas verdaderas para el entrenamiento.

        Returns:
        None
        """
        try:
            self.logger.info(f"Entrenando modelo {self.model.__class__.__name__}...")
            self.model.train(X, y)
            self.logger.info("Modelo entrenado con éxito.")
        except Exception as e:
            self.logger.error(f"Error al entrenar el modelo: {e}")

    def evaluate(self, X, y):
        """
        Evalúa el modelo con los datos proporcionados.

        Parameters:
        X (array-like): Datos de entrada para la evaluación.
        y (array-like): Etiquetas verdaderas para la evaluación.

        Returns:
        tuple: MAE y RMSE del modelo.
        """
        try:
            mae, rmse = self.model.evaluate(X, y)
            self.logger.info(f"MAE: {mae:.2f}")
            self.logger.info(f"RMSE: {rmse:.2f}")
            return mae, rmse
        except Exception as e:
            self.logger.error(f"Error al evaluar el modelo: {e}")
            return None, None

    def fine_tune(self, X, y):
        """
        Ejecuta ajuste de hiperparámetros para el modelo.

        Parameters:
        X (array-like): Datos de entrada para el ajuste.
        y (array-like): Etiquetas verdaderas para el ajuste.

        Returns:
        dict: Mejores parámetros encontrados.
        """
        try:
            if hasattr(self.model, "get_param_grid"):
                param_grid = self.model.get_param_grid()
                if param_grid is None:
                    self.logger.info(f"El modelo {self.model.__class__.__name__} no requiere fine-tuning.")
                    return None
            else:
                self.logger.warning(f"El modelo {self.model.__class__.__name__} no tiene una configuración de fine-tuning.")
                return None

            if hasattr(self.model, "fine_tune"):
                best_params = self.model.fine_tune(X, y, param_grid)
                self.logger.info(f"Mejores parámetros encontrados: {best_params}")
                return best_params
            else:
                self.logger.warning(f"El modelo {self.model.__class__.__name__} no soporta fine-tuning.")
                return None
        except Exception as e:
            self.logger.error(f"Error al ajustar los hiperparámetros: {e}")
            return None