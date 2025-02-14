from src.models.factory import ModelFactory
from src.pipeline.trainer import Trainer
from src.utils.interfaces.forecasting_pipeline.model_trainer_interface import ModelTrainerInterface

class ModelTrainer(ModelTrainerInterface):
    def train_and_evaluate(self, model_name, df_original):
        """
        Entrena y evalúa un modelo específico, devolviendo sus métricas y predicciones.

        Parameters:
        model_name (str): Nombre del modelo a entrenar y evaluar.
        df_original (DataFrame): Datos originales para entrenamiento y evaluación.

        Returns:
        dict: Métricas y predicciones del modelo.
        """
        try:
            print(f"\n🏆 Probando modelo: {model_name} 🏆")
            
            model = ModelFactory.get_model(model_name)
            df = model.load_data(df_original)  # Eliminar las últimas dos semanas antes de entrenar
            df = df.iloc[:-2]

            train_size = int(len(df) * 0.8)
            train, test = df.iloc[:train_size], df.iloc[train_size:]

            X_train, y_train = train.drop(columns=["y"]), train["y"]
            X_test, y_test = test.drop(columns=["y"]), test["y"]

            trainer = Trainer(model)
            trainer.fine_tune(X_train, y_train)

            # Entrenar modelo con los mejores hiperparámetros
            trainer.train(X_train, y_train)
            mae, rmse = trainer.evaluate(X_test, y_test)

            # Guardar métricas en resultados
            # 🚀 Predecir los próximos 2 puntos (semanas 29 y 30)
            df_futuro = df_original.iloc[-2:].copy()
            df_futuro = model.load_data(df_futuro, train_data=df_original)  # Transformar datos como en entrenamiento
            X_futuro = df_futuro.drop(columns=["y"])
            print("📌 Prediciendo las próximas 2 semanas...")
            predicciones = model.predict(X_futuro)
            print(f"📌 Predicciones para semanas 29 y 30: {predicciones}")

            return {"MAE": mae, "RMSE": rmse, "Predicciones": predicciones.tolist()}
        except KeyError as e:
            print(f"Error: Columna no encontrada - {e}")
        except Exception as e:
            print(f"Error al entrenar y evaluar el modelo {model_name}: {e}")