from abc import ABC, abstractmethod

class VisualizationForecastingInterface(ABC):
    """
    Interfaz para la visualización de resultados de forecasting.

    Methods:
    save_results(resultados, df_original):
        Genera visualizaciones y guarda los resultados en CSV.
    """
    @abstractmethod
    def save_results(self, resultados, df_original):
        """
        Genera visualizaciones y guarda los resultados en CSV.

        Parameters:
        resultados (dict): Diccionario con los resultados de las predicciones de los modelos.
        df_original (DataFrame): DataFrame con los datos originales.

        Returns:
        None
        """
        pass