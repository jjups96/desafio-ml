import pandas as pd
from src.utils.interfaces.tagging.results_saver_interface import ResultsSaverTagInterface

class CSVResultsSaver(ResultsSaverTagInterface):
    def save(self, results, filepath):
        """
        Guarda los resultados de clasificación en un archivo CSV.

        Parameters:
        results (list): Lista de diccionarios con los resultados de la clasificación.
        filepath (str): Ruta del archivo donde se guardarán los resultados.

        Returns:
        DataFrame: DataFrame con los resultados guardados.
        """
        try:
            df_results = pd.DataFrame([
                {"Sentence": r["sentence"], "Best Label": r["best_label"], **r["predictions"]}
                for r in results
            ])
            df_results.to_csv(filepath, index=False)
            return df_results
        except Exception as e:
            print(f"Error al guardar los resultados: {e}")
            return pd.DataFrame()