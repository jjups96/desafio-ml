from abc import ABC, abstractmethod

class VisualizationTagInterface(ABC):
    """
    Interfaz para la visualización de resultados de clasificación de etiquetas.

    Methods:
    plot(filepath):
        Visualiza los resultados de clasificación desde el archivo especificado.
    """
    @abstractmethod
    def plot(self, filepath):
        """
        Visualiza los resultados de clasificación desde el archivo especificado.

        Parameters:
        filepath (str): Ruta del archivo CSV con los resultados de clasificación.

        Returns:
        None
        """
        pass