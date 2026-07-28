import spacy
import time
import subprocess

class NERMasking:
    """Privacy entity masking using named entity recognition (adapted to ECHR entity types)"""
    
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_lg")
        except OSError:
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_lg"], check=True)
            self.nlp = spacy.load("en_core_web_lg")
        
      
        self.entity_mapping = {
            'PERSON': '[PERSON]',        # Name
            'CODE': '[CODE]',            # Identification number or code
            'LOC': '[LOCATION]',         # Location 
            'ORG': '[ORGANIZATION]',     # Organization
            'DEM': '[DEMOGRAPHIC]',      # Demographic attributes
            'DATETIME': '[DATETIME]',    # Date, time
            'QUANTITY': '[QUANTITY]',    # Quantity
            'MISC': '[MISC]'             # Other
        }
        
       
        self.generalization_rules = {
            'DATETIME': {
                'year': lambda x: '[YEAR]',
                'month': lambda x: '[MONTH]',
                'day': lambda x: '[DAY]',
                'full': lambda x: '[DATETIME]'
            },
            'DEM': {
                'age': lambda x: '[ADULT_AGE]' if int(x) >= 18 else '[MINOR_AGE]',
                'default': lambda x: '[DEMOGRAPHIC]'
            }
        }
    
    def _generalize_entity(self, entity_text, entity_label):
       
        if entity_label == 'DATETIME':
            if len(entity_text) == 4 and entity_text.isdigit():  
                return self.generalization_rules['DATETIME']['year'](entity_text)
            elif '/' in entity_text or '-' in entity_text:  
                return self.generalization_rules['DATETIME']['full'](entity_text)
        elif entity_label == 'DEM':
            if 'year' in entity_text.lower() or 'age' in entity_text.lower():
                age = ''.join(filter(str.isdigit, entity_text))
                return self.generalization_rules['DEM']['age'](age) if age else self.generalization_rules['DEM']['default'](entity_text)
            return self.generalization_rules['DEM']['default'](entity_text)
        
        return self.entity_mapping.get(entity_label, f'[ENTITY_{entity_label}]')
    
    def mask(self, text):
        
        start_time = time.time()
        
        doc = self.nlp(text)
        masked_text = text
        
        
        for ent in sorted(doc.ents, key=lambda x: len(x.text), reverse=True):
           
            spacy_to_echr = {
                "PERSON": "PERSON",
                "ORG": "ORG",
                "GPE": "LOC",
                "DATE": "DATETIME",
                "TIME": "DATETIME",
                "MONEY": "QUANTITY",
                "CARDINAL": "QUANTITY",
                "ORDINAL": "QUANTITY",
                "FAC": "LOC",
                "LAW": "MISC"
            }
            echr_label = spacy_to_echr.get(ent.label_, None)
            if echr_label and echr_label in self.entity_mapping:
                replacement = self._generalize_entity(ent.text, echr_label)
                masked_text = masked_text.replace(ent.text, replacement)
        
        processing_time = time.time() - start_time
        return masked_text, processing_time
    
    def get_supported_entities(self):
        
        return list(self.entity_mapping.keys())