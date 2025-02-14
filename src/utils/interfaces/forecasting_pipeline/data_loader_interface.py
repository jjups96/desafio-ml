from abc import ABC, abstractmethod

class DataLoaderInterface(ABC):
    """
    Interfaz para cargar datos de forecasting.

    Methods:
    load_data(file_path):
        Carga los datos desde la ruta especificada.
    """
    @abstractmethod
    def load_data(self, file_path):
        """
        Carga los datos desde la ruta especificada.

        Parameters:
        file_path (str): Ruta del archivo de datos.

        Returns:
        DataFrame: Datos cargados en un DataFrame.
        """
        pass