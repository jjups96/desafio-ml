import os
from src.models.zeroshoot_classifier import ZeroShotClassifier
from src.utils.results_saver import CSVResultsSaver
from src.utils.visualization import TagResultsPlotter

def create_results_dir(directory):
    """
    Crea el directorio de resultados si no existe.

    Parameters:
    directory (str): Ruta del directorio a crear.

    Returns:
    None
    """
    os.makedirs(directory, exist_ok=True)

def classify_sentences(classifier, sentences, labels):
    """
    Clasifica las sentencias usando el clasificador proporcionado.

    Parameters:
    classifier (ZeroShotClassifier): Instancia del clasificador.
    sentences (list): Lista de sentencias a clasificar.
    labels (list): Lista de etiquetas posibles.

    Returns:
    list: Resultados de la clasificación.
    """
    try:
        return classifier.predict(sentences, labels)
    except Exception as e:
        print(f"Error al clasificar las sentencias: {e}")
        return []

def main(classifier, results_saver, plotter):
    """
    Función principal para ejecutar el pipeline de clasificación de etiquetas.

    Parameters:
    classifier (ZeroShotClassifier): Instancia del clasificador.
    results_saver (CSVResultsSaver): Instancia del guardador de resultados.
    plotter (TagResultsPlotter): Instancia del visualizador de resultados.

    Returns:
    None
    """
    RESULTS_DIR = "results"
    create_results_dir(RESULTS_DIR)

    sentences = [
        "The axolotl is more than a peculiar amphibian; in its natural environment, it plays an essential role in the ecological stability of the Xochimilco canals.",
        "Geoffrey Hinton, Yann LeCun and Yoshua Bengio are considered the ‘godfathers’ of an essential technique in artificial intelligence, called ‘deep learning’.",
        "Greenland is about to open up to adventure-seeking visitors. How many tourists will come is yet to be seen, but the three new airports will bring profound change.",
        "GitHub Copilot is an AI coding assistant that helps you write code faster and with less effort, allowing you to focus more energy on problem solving and collaboration.",
        "I have a problem with my laptop that needs to be resolved asap!!"
    ]

    labels = ["urgent", "artificial intelligence", "computer", "travel", "animal", "fiction"]

    results = classify_sentences(classifier, sentences, labels)

    if not results:
        print("No se pudieron clasificar las sentencias.")
        return

    results_filepath = os.path.join(RESULTS_DIR, "tag_class_zero_shoot_results.csv")
    try:
        df_results = results_saver.save(results, results_filepath)
    except Exception as e:
        print(f"Error al guardar los resultados: {e}")
        return

    print("\n📜 Resultados guardados en results/tag_class_zero_shoot_results.csv")
    print(df_results)

    try:
        plotter.plot(results_filepath)
    except Exception as e:
        print(f"Error al generar la visualización de los resultados: {e}")

if __name__ == "__main__":
    classifier = ZeroShotClassifier()
    results_saver = CSVResultsSaver()
    plotter = TagResultsPlotter()
    main(classifier, results_saver, plotter)