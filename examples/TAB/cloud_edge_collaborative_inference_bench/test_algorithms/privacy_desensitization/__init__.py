from .regex_pseudonymization import regex_pseudonymization
from .ner_masking import ner_masking_with_generalization
from .differential_privacy import apply_differential_privacy
from .privacy_evaluator import PrivacyEvaluator

class PrivacyPipeline:
    def __init__(self, methods=None, nlp_model=None, similarity_model=None):
        self.methods = methods or ["regex", "ner", "dp"]
        self.nlp_model = nlp_model
        self.similarity_model = similarity_model

    def desensitize(self, text, embeddings=None):
        current_text = text
        current_embeddings = embeddings
        if "regex" in self.methods:
            current_text = regex_pseudonymization(current_text)
        if "ner" in self.methods:
            current_text = ner_masking_with_generalization(current_text, nlp_model=self.nlp_model)
        if "dp" in self.methods and current_embeddings is not None:
            current_embeddings = apply_differential_privacy(current_embeddings)
        return current_text, current_embeddings


