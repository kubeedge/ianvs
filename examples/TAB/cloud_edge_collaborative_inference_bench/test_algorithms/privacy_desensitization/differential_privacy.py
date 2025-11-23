<<<<<<< HEAD
import numpy as np
import time
from diffprivlib.mechanisms import Laplace
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import os

class DifferentialPrivacy:
    """Implementing differential privacy protection methods"""
    
    def __init__(self, epsilon=1.0, model_path: str | None = None):

        self.epsilon = epsilon
        self.laplace = Laplace(epsilon=epsilon, sensitivity=1)


        if model_path is None:
            model_path = os.environ.get(
                "IANVS_LOCAL_MODEL",
                os.path.abspath(
                    os.path.join(
                        os.path.dirname(__file__),
                        "../../local_model",
                    )
                ),
            )

    
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.classifier = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
        )
    
    def _identify_sensitive_regions(self, text):
      
        # Divide the text into sentences and identify sentences that may contain sensitive information
        sentences = text.split('. ')
        sensitive_regions = []
        
        for i, sentence in enumerate(sentences):
            if len(sentence) == 0:
                continue
                
            result = self.classifier(sentence)[0]
            if result['label'] == 'NEGATIVE' or result['score'] > 0.7:
                sensitive_regions.append(i)
                
        return sensitive_regions
    
    def add_noise(self, text):
        """Add calibration noise to text for differential privacy"""
        start_time = time.time()
        
        sentences = text.split('. ')
        sensitive_regions = self._identify_sensitive_regions(text)
        
    
        for i in sensitive_regions:
            if i >= len(sentences):
                continue
                
           
            words = sentences[i].split()
            if len(words) > 3:  
                num_changes = max(1, int(len(words) * 0.1))  
                for _ in range(num_changes):
                    idx = np.random.randint(0, len(words))
                    if np.random.laplace(loc=0, scale=1/self.epsilon) > 0.5:
                        words[idx] = f"[WORD_{np.random.randint(1000)}]"
                
                sentences[i] = ' '.join(words)
        
        processed_text = '. '.join(sentences)
        processing_time = time.time() - start_time
        
        return processed_text, processing_time
    
    def adjust_epsilon(self, epsilon):
        """Adjust Privacy Budget"""
        self.epsilon = epsilon
        self.laplace = Laplace(epsilon=epsilon, sensitivity=1)
=======
version https://git-lfs.github.com/spec/v1
oid sha256:93a72c4dee0ff97775c302e9a0e06c156d4df1615d0d013125290b20940bb4b4
size 2825
>>>>>>> 9676c3e (ya toh aar ya toh par)
