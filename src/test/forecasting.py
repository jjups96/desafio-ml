import unittest
from src.utils.modules.data_loader import DataLoader
from src.utils.modules.model_trainer import ModelTrainer

class TestDataLoader(unittest.TestCase):
    def test_load_data(self):
        """
        Prueba la carga de datos y la eliminación de la columna 'Product'.
        """
        data_loader = DataLoader()
        df = data_loader.load_data("data/sales.csv")
        self.assertFalse(df.empty)
        self.assertNotIn("Product", df.columns)

class TestModelTrainer(unittest.TestCase):
    def test_train_and_evaluate(self):
        """
        Prueba el entrenamiento y evaluación del modelo.
        """
        model_trainer = ModelTrainer()
        data_loader = DataLoader()
        df = data_loader.load_data("data/sales.csv")
        result = model_trainer.train_and_evaluate("XGBoost", df)
        self.assertIn("MAE", result)
        self.assertIn("RMSE", result)
        self.assertIn("Predicciones", result)

if __name__ == "__main__":
    unittest.main()