from sklearn.feature_extraction.text import TfidfVectorizer

class FeatureExtractor:
    """Convierte texto en embeddings TF-IDF."""

    def __init__(self):
        """
        Inicializa el extractor de características TF-IDF.
        """
        self.vectorizer = TfidfVectorizer()

    def fit_transform(self, texts):
        """
        Ajusta el vectorizador TF-IDF y transforma los textos.

        Parameters:
        texts (list): Lista de textos a transformar.

        Returns:
        sparse matrix: Matriz dispersa con los embeddings TF-IDF.
        """
        try:
            return self.vectorizer.fit_transform(texts)
        except Exception as e:
            print(f"Error al ajustar y transformar los textos: {e}")
            return None

    def transform(self, texts):
        """
        Transforma los textos usando el vectorizador TF-IDF ajustado.

        Parameters:
        texts (list): Lista de textos a transformar.

        Returns:
        sparse matrix: Matriz dispersa con los embeddings TF-IDF.
        """
        try:
            return self.vectorizer.transform(texts)
        except Exception as e:
            print(f"Error al transformar los textos: {e}")
            return None