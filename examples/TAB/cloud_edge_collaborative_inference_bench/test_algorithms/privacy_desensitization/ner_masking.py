from typing import Optional

def generalize_address(address: str) -> str:
    return "[LOCATION]"

def ner_masking_with_generalization(text: str, nlp_model: Optional[object] = None) -> str:
    if nlp_model is None:
        return text
    doc = nlp_model(text)
    masked = text
    for ent in doc.ents:
        if ent.label_ == 'PERSON':
            masked = masked.replace(ent.text, '[PERSON]')
        elif ent.label_ in ('GPE', 'LOC', 'ADDRESS'):
            masked = masked.replace(ent.text, generalize_address(ent.text))
        elif ent.label_ in ('ORG',):
            masked = masked.replace(ent.text, '[ORG]')
        elif ent.label_ in ('DATE', 'TIME'):
            masked = masked.replace(ent.text, '[TIME]')
    return masked


