import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.functional as F

from src.models.base_tag_classifier_model import ClassifierInterface
from transformers import pipeline

class ZeroShotClassifier(ClassifierInterface):
    """Clasificador de texto usando Zero-Shot Classification con Hugging Face."""

    def __init__(self, model_name="facebook/bart-large-mnli"):
        self.classifier = pipeline("zero-shot-classification", model=model_name)

    def predict(self, sentences, labels):
        """Clasifica sentencias con probabilidades usando Zero-Shot Classification."""
        results = self.classifier(sentences, candidate_labels=labels, multi_label=True)

        predictions = []
        for i, sentence in enumerate(sentences):
            label_scores = dict(zip(results[i]["labels"], results[i]["scores"]))

            # Encontrar la etiqueta con mayor probabilidad
            best_label = max(label_scores, key=label_scores.get)

            predictions.append({"sentence": sentence, "best_label": best_label, "predictions": label_scores})

        return predictions