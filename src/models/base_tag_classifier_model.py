from abc import ABC, abstractmethod

class ClassifierInterface(ABC):
    @abstractmethod
    def predict(self, sentences, labels):
        pass