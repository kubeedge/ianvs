<<<<<<< HEAD
from .regex_pseudonymization import RegexPseudonymization
from .ner_masking import NERMasking
from .differential_privacy import DifferentialPrivacy

def desensitize(text, methods=None):
    
    methods = methods or ["regex"]
    current_text = text

    regex_processor = RegexPseudonymization()
    ner_processor = NERMasking()
    dp_processor = DifferentialPrivacy()

    if "regex" in methods:
        current_text, _ = regex_processor.anonymize(current_text)
    if "ner" in methods:
        current_text, _ = ner_processor.mask(current_text)
    if "dp" in methods:
        current_text, _ = dp_processor.add_noise(current_text)

    return current_text
=======
version https://git-lfs.github.com/spec/v1
oid sha256:558fb285f717b587f62b4dd67710df1e0a09bc166576ccc49f06f265b2181afe
size 660
>>>>>>> 9676c3e (ya toh aar ya toh par)
