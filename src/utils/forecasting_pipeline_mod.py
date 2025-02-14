import os
from src.utils.modules.data_loader import DataLoader
from src.utils.modules.model_trainer import ModelTrainer
from src.utils.visualization import Visualization

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_forecasting_pipeline(data_file, data_loader, model_trainer, visualization):
    """
    Ejecuta el pipeline de forecasting para todos los modelos.

    Parameters:
    data_file (str): Ruta al archivo de datos.
    data_loader (DataLoader, optional): Instancia de DataLoader. Si no se proporciona, se crea una nueva.
    model_trainer (ModelTrainer, optional): Instancia de ModelTrainer. Si no se proporciona, se crea una nueva.
    visualization (Visualization, optional): Instancia de Visualization. Si no se proporciona, se crea una nueva.

    Returns:
    None
    """
    modelos = ["XGBoost", "Prophet", "SARIMA", "Ridge"]
    resultados = {}

    # Crear instancias si no se proporcionan
    data_loader = data_loader or DataLoader()
    model_trainer = model_trainer or ModelTrainer()
    visualization = visualization or Visualization()

    try:
        df_original = data_loader.load_data(data_file)
    except Exception as e:
        print(f"Error al cargar los datos: {e}")
        return

    for model_name in modelos:
        try:
            resultados[model_name] = model_trainer.train_and_evaluate(model_name, df_original.copy())
        except Exception as e:
            print(f"Error al entrenar y evaluar el modelo {model_name}: {e}")
            resultados[model_name] = None

    try:
        visualization.save_results(resultados, df_original)
    except Exception as e:
        print(f"Error al guardar los resultados: {e}")