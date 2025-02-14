import re

class TextPreprocessor:
    """Limpia y normaliza texto."""
    
    def clean_text(self, text):
        return re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
