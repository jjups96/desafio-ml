from abc import ABC, abstractmethod

class ResultsSaverTagInterface(ABC):
    """
    Interfaz para guardar resultados de clasificación de etiquetas.

    Methods:
    save(results, filepath):
        Guarda los resultados de clasificación en la ruta especificada.
    """
    @abstractmethod
    def save(self, results, filepath):
        """
        Guarda los resultados de clasificación en la ruta especificada.

        Parameters:
        results (dict): Diccionario con los resultados de la clasificación.
        filepath (str): Ruta del archivo donde se guardarán los resultados.

        Returns:
        None
        """
        pass