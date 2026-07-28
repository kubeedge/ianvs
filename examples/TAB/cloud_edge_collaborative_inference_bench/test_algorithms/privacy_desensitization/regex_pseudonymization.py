import re
import time

class RegexPseudonymization:
    """Use regular expressions for privacy data anonymization"""
    
    def __init__(self):
        self.patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b(?:\+\d{1,3}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b',
            'id': r'\b(?:\d{3}-\d{2}-\d{4}|\d{10,12}|\w{2}\d{6}|[A-Z0-9]{5,10})\b',  
            'name': r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b',  
            'address': r'\b\d+\s[A-Za-z\s,]+(?:St|Ave|Rd|Blvd|Ln|Street|Avenue)\b', 
            'zipcode': r'\b\d{5}(?:-\d{4})?\b',
            'date': r'\b(?:\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2}|[A-Z][a-z]+\s\d{1,2},\s\d{4})\b',  
            'age': r'\b\d{1,3}\s*(?:years? old|y\.o\.?|age)\b',  
            'code': r'\b[A-Z0-9\-/]{5,15}\b',  
            'quantity': r'\b\d+(?:\.\d+)?\s*(?:%|\$|€|£|kg|m|cm)\b'  
        }
    
    
    
        self.replacements = {
            'email': '[CODE]',
            'phone': '[CODE]',
            'id': '[CODE]',
            'name': '[PERSON]',
            'address': '[LOC]',
            'zipcode': '[CODE]',
            'date': '[DATETIME]',
            'age': '[DEM]',
            'code': '[CODE]',
            'quantity': '[QUANTITY]'
        }
    
    def anonymize(self, text):
       
        start_time = time.time()
        
        
        for category, pattern in self.patterns.items():
            text = re.sub(pattern, self.replacements[category], text, flags=re.IGNORECASE)
        
        processing_time = time.time() - start_time
        return text, processing_time
    
    def get_pii_categories(self):
        
        return list(self.patterns.keys())