import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

from src.utils.interfaces.tagging.visualization_interface import VisualizationTagInterface
from src.utils.interfaces.forecasting_pipeline.visualization_interface import VisualizationForecastingInterface

# Crear directorio si no existe
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def plot_predictions(df_original, resultados, save=True):
    """
    Visualiza y guarda las predicciones de los modelos contra las ventas históricas.

    Parameters:
    df_original (DataFrame): DataFrame con los datos originales.
    resultados (dict): Diccionario con los resultados de las predicciones de los modelos.
    save (bool): Indica si se debe guardar la visualización.

    Returns:
    None
    """
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=df_original.index, y=df_original["Sales"], label="Ventas Históricas", linestyle="dotted", color="black")

    # Agregar predicciones de cada modelo
    for modelo, metricas in resultados.items():
        if "Predicciones" in metricas:
            semanas_futuras = [len(df_original)-2, len(df_original)-1]
            plt.plot(semanas_futuras, metricas["Predicciones"], marker="o", label=f"Predicción {modelo}")

    plt.xlabel("Semana")
    plt.ylabel("Ventas")
    plt.title("Predicciones de Modelos vs. Ventas Históricas")
    plt.legend()
    
    if save:
        plt.savefig(os.path.join(RESULTS_DIR, "forecasting_plot_predictions.png"))
    
    plt.show()

def plot_errors(resultados, save=True):
    """
    Muestra y guarda una comparación de errores (MAE y RMSE) entre modelos.

    Parameters:
    resultados (dict): Diccionario con los resultados de las predicciones de los modelos.
    save (bool): Indica si se debe guardar la visualización.

    Returns:
    None
    """
    plt.figure(figsize=(8, 5))
    df_resultados = pd.DataFrame(resultados).T
    df_resultados[["MAE", "RMSE"]].plot(kind="bar", figsize=(8, 5))
    plt.title("Comparación de Errores (MAE y RMSE)")
    plt.ylabel("Error")
    plt.xticks(rotation=0)
    plt.legend(["MAE", "RMSE"])
    
    if save:
        plt.savefig(os.path.join(RESULTS_DIR, "forecasting_plot_errors.png"))
    
    plt.show()

def plot_tag_results(csv_path=os.path.join(RESULTS_DIR, "classification_results.csv")):
    """
    Visualiza las probabilidades de clasificación en un heatmap.

    Parameters:
    csv_path (str): Ruta al archivo CSV con los resultados de clasificación.

    Returns:
    None
    """
    try:
        # Cargar datos
        df = pd.read_csv(csv_path)

        # Extraer la mejor etiqueta
        best_labels = df[["Sentence", "Best Label"]]

        # Preparar datos para el heatmap
        df_melted = df.melt(id_vars=["Sentence", "Best Label"], var_name="Category", value_name="Probability")

        # Convertir "Sentence" a un índice categórico para mantener el orden original
        df_melted["Sentence"] = pd.Categorical(df_melted["Sentence"], categories=df["Sentence"], ordered=True)

        # Crear pivot para el heatmap sin cambiar el orden original
        heatmap_data = df_melted.pivot(index="Sentence", columns="Category", values="Probability")

        plt.figure(figsize=(10, 6))

        # Dibujar el heatmap
        ax = sns.heatmap(heatmap_data, annot=True, cmap="Blues", fmt=".2f")

        # Agregar la mejor etiqueta en el eje Y sin cambiar el orden original
        y_labels = [f"{sentence}\n(Best: {best_label})" for sentence, best_label in zip(best_labels["Sentence"], best_labels["Best Label"])]
        ax.set_yticklabels(y_labels, rotation=0)

        plt.title("Probabilidad de Clasificación por Categoría")
        plt.ylabel("Sentencia")
        plt.xlabel("Categoría")

        # Guardar gráfico
        plt.savefig(os.path.join(RESULTS_DIR, "tag_classification_heatmap.png"))
        plt.show()
    except Exception as e:
        print(f"Error al generar el heatmap de clasificación: {e}")

def save_results(resultados, df_original):
    """
    Genera visualizaciones y guarda los resultados en CSV.

    Parameters:
    resultados (dict): Diccionario con los resultados de las predicciones de los modelos.
    df_original (DataFrame): DataFrame con los datos originales.

    Returns:
    None
    """
    try:
        # 🏆 Determinar el mejor modelo basado en RMSE
        mejor_modelo = min(resultados, key=lambda x: resultados[x]["RMSE"])

        print("\n📊 RESULTADOS FINALES:")
        for modelo, metricas in resultados.items():
            print(f"📌 {modelo}: MAE={metricas['MAE']:.2f}, RMSE={metricas['RMSE']:.2f}, Predicciones={metricas['Predicciones']}")

        print(f"\n🥇 ¡El mejor modelo es {mejor_modelo} con RMSE={resultados[mejor_modelo]['RMSE']:.2f}! 🏆")

        # 📊 Generar visualizaciones
        plot_predictions(df_original, resultados)
        plot_errors(resultados)

        # 📜 Guardar resultados en CSV
        df_resultados = pd.DataFrame(resultados).T
        df_resultados.to_csv(os.path.join(RESULTS_DIR, "forecasting_results.csv"))

        print(f"\n✅ Resultados guardados en {RESULTS_DIR}/forecasting_results.csv")
        print(f"✅ Gráficos guardados en {RESULTS_DIR}/")
    except Exception as e:
        print(f"Error al guardar los resultados: {e}")

class TagResultsPlotter(VisualizationTagInterface):
    def plot(self, filepath):
        """
        Visualiza las probabilidades de clasificación en un heatmap.

        Parameters:
        filepath (str): Ruta al archivo CSV con los resultados de clasificación.

        Returns:
        None
        """
        try:
            # Cargar datos
            df = pd.read_csv(filepath)

            # Extraer la mejor etiqueta
            best_labels = df[["Sentence", "Best Label"]]

            # Preparar datos para el heatmap
            df_melted = df.melt(id_vars=["Sentence", "Best Label"], var_name="Category", value_name="Probability")

            # Convertir "Sentence" a un índice categórico para mantener el orden original
            df_melted["Sentence"] = pd.Categorical(df_melted["Sentence"], categories=df["Sentence"], ordered=True)

            # Crear pivot para el heatmap sin cambiar el orden original
            heatmap_data = df_melted.pivot(index="Sentence", columns="Category", values="Probability")

            plt.figure(figsize=(10, 6))

            # Dibujar el heatmap
            ax = sns.heatmap(heatmap_data, annot=True, cmap="Blues", fmt=".2f")

            # Agregar la mejor etiqueta en el eje Y sin cambiar el orden original
            y_labels = [f"{sentence}\n(Best: {best_label})" for sentence, best_label in zip(best_labels["Sentence"], best_labels["Best Label"])]
            ax.set_yticklabels(y_labels, rotation=0)

            plt.title("Probabilidad de Clasificación por Categoría")
            plt.ylabel("Sentencia")
            plt.xlabel("Categoría")

            # Guardar gráfico
            plt.savefig(os.path.join(RESULTS_DIR, "tag_classification_heatmap.png"))
            plt.show()
        except Exception as e:
            print(f"Error al generar el heatmap de clasificación: {e}")

class Visualization(VisualizationForecastingInterface):
    def save_results(self, resultados, df_original):
        """
        Genera visualizaciones y guarda los resultados en CSV.

        Parameters:
        resultados (dict): Diccionario con los resultados de las predicciones de los modelos.
        df_original (DataFrame): DataFrame con los datos originales.

        Returns:
        None
        """
        try:
            # 🏆 Determinar el mejor modelo basado en RMSE
            mejor_modelo = min(resultados, key=lambda x: resultados[x]["RMSE"])

            print("\n📊 RESULTADOS FINALES:")
            for modelo, metricas in resultados.items():
                print(f"📌 {modelo}: MAE={metricas['MAE']:.2f}, RMSE={metricas['RMSE']:.2f}, Predicciones={metricas['Predicciones']}")

            print(f"\n🥇 ¡El mejor modelo es {mejor_modelo} con RMSE={resultados[mejor_modelo]['RMSE']:.2f}! 🏆")

            # 📊 Generar visualizaciones
            plot_predictions(df_original, resultados)
            plot_errors(resultados)

            # 📜 Guardar resultados en CSV
            df_resultados = pd.DataFrame(resultados).T
            df_resultados.to_csv(os.path.join(RESULTS_DIR, "forecasting_results.csv"))

            print(f"\n✅ Resultados guardados en {RESULTS_DIR}/forecasting_results.csv")
            print(f"✅ Gráficos guardados en {RESULTS_DIR}/")
        except Exception as e:
            print(f"Error al guardar los resultados: {e}")