from src.utils.forecasting_pipeline_mod import run_forecasting_pipeline
from src.utils.modules.data_loader import DataLoader
from src.utils.modules.model_trainer import ModelTrainer
from src.utils.visualization import Visualization

def main():
    """
    Función principal para ejecutar el pipeline de forecasting.

    Parameters:
    None

    Returns:
    None
    """
    data_file = "data/sales.csv"  # Especifica la ruta a tu archivo de datos
    data_loader = DataLoader()
    model_trainer = ModelTrainer()
    visualization = Visualization()
    try:
        run_forecasting_pipeline(data_file, data_loader, model_trainer, visualization)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()