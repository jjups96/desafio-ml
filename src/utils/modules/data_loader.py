import pandas as pd
from src.utils.interfaces.forecasting_pipeline.data_loader_interface import DataLoaderInterface

class DataLoader(DataLoaderInterface):
    def load_data(self, file_path):
        """
        Carga el dataset original y lo prepara eliminando la columna 'Product'.

        Parameters:
        file_path (str): Ruta del archivo de datos.

        Returns:
        DataFrame: Datos cargados y preparados.
        """
        try:
            df = pd.read_csv(file_path)
            return df.drop(columns=["Product"])
        except FileNotFoundError:
            print(f"Error: El archivo {file_path} no fue encontrado.")
        except pd.errors.EmptyDataError:
            print(f"Error: El archivo {file_path} está vacío.")
        except KeyError:
            print(f"Error: La columna 'Product' no existe en el archivo {file_path}.")
        except Exception as e:
            print(f"Error al cargar los datos: {e}")