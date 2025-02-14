from abc import ABC, abstractmethod

class ModelTrainerInterface(ABC):
    """
    Interfaz para entrenar y evaluar modelos de forecasting.

    Methods:
    train_and_evaluate(model_name, df_original):
        Entrena y evalúa el modelo especificado.
    """
    @abstractmethod
    def train_and_evaluate(self, model_name, df_original):
        """
        Entrena y evalúa el modelo especificado.

        Parameters:
        model_name (str): Nombre del modelo a entrenar y evaluar.
        df_original (DataFrame): Datos originales para entrenamiento y evaluación.

        Returns:
        dict: Métricas y predicciones del modelo.
        """
        pass