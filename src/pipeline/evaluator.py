from sklearn.metrics import classification_report

class Evaluator:
    """Evalúa un modelo de clasificación."""

    def evaluate(self, model, X, y):
        """
        Evalúa el modelo de clasificación y genera un reporte de clasificación.

        Parameters:
        model: Modelo de clasificación a evaluar.
        X (array-like): Datos de entrada para las predicciones.
        y (array-like): Etiquetas verdaderas para las predicciones.

        Returns:
        str: Reporte de clasificación.
        """
        try:
            predictions = model.predict(X)
            return classification_report(y, predictions)
        except Exception as e:
            print(f"Error al evaluar el modelo: {e}")
            return ""