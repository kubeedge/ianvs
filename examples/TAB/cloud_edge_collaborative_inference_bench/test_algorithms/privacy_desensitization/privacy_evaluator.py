import numpy as np
import spacy
from sentence_transformers import SentenceTransformer, util
import os
from sedna.datasources import BaseDataSource
import json

class PrivacyEvaluator:
    
    def __init__(self, dataset: BaseDataSource = None):
        # Force offline mode for transformers/sentence-transformers to avoid HF downloads
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        # Prefer local SentenceTransformer directory if provided
        local_st_model_dir = os.environ.get(
            "ST_MODEL_DIR",
            "./examples/TAB/cloud_edge_collaborative_inference_bench/all-MiniLM-L6-v2",
        )

        self.nlp = spacy.load("en_core_web_lg")
        # Load embeddings model from local directory
        self.sentence_model = SentenceTransformer(local_st_model_dir)
        self.echr_dataset = self._parse_dataset(dataset) if dataset else None
        self.echr_pii_patterns = self._extract_echr_pii_patterns() if self.echr_dataset else set()
    
    def set_dataset(self, dataset: BaseDataSource):
        self.echr_dataset = self._parse_dataset(dataset)
        self.echr_pii_patterns = self._extract_echr_pii_patterns()
    
    def _parse_dataset(self, dataset: BaseDataSource) -> list:
        def parse_question_embedded(q):
            text, annotations = q, {}
            if not isinstance(q, str):
                return text, annotations
            marker = "\n\nANNOTATIONS_JSON="
            if marker in q:
                head, tail = q.rsplit(marker, 1)
                text = head.strip()
                try:
                    annotations = json.loads(tail.strip())
                except Exception:
                    annotations = {}
                return text, annotations
            
            if "<ANN>" in q and "</ANN>" in q:
                try:
                    head, rest = q.split("<ANN>", 1)
                    json_str, _ = rest.split("</ANN>", 1)
                    text = head.strip()
                    annotations = json.loads(json_str.strip())
                except Exception:
                    annotations = {}
                return text, annotations
            return text, annotations

        parsed = []
        for x, y in zip(dataset.x, dataset.y):
            doc = {}
            if isinstance(x, dict):
                if "question" in x:
                    text_val, ann = parse_question_embedded(x.get("question", ""))
                    doc = {
                        "text": text_val,
                        "annotations": ann or x.get("annotations", {}) or (y.get("annotations", {}) if isinstance(y, dict) else {}),
                        "doc_id": x.get("doc_id")
                    }
                else:
                    doc = {
                        "text": x.get("text", ""),
                        "annotations": x.get("annotations", {}) or (y.get("annotations", {}) if isinstance(y, dict) else {}),
                        "doc_id": x.get("doc_id")
                    }
            else:
                text = str(x) if not isinstance(x, str) else x
                text_val, ann = parse_question_embedded(text)
                if not ann and isinstance(y, dict):
                    ann = y.get("annotations", {})
                doc = {
                    "text": text_val,
                    "annotations": ann,
                    "doc_id": (y.get("doc_id") if isinstance(y, dict) else None)
                }
            parsed.append(doc)
        return parsed
    
    def _extract_echr_pii_patterns(self):
        if not self.echr_dataset:
            raise ValueError("The dataset has not been initialized。")
        
        patterns = set()
        for doc in self.echr_dataset:
            for annotator in doc.get("annotations", {}).values():
                for entity in annotator.get("entity_mentions", []):
                    if entity.get("identifier_type") in ["DIRECT", "QUASI"]:
                        patterns.add((entity["span_text"], entity["entity_type"]))
        return patterns
    
    def detect_pii_entities(self, text):
        doc = self.nlp(text)
        detected = []
        
       
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
        
     
        for ent in doc.ents:
            if ent.label_ in spacy_to_echr:
                detected.append({
                    "text": ent.text, 
                    "label": spacy_to_echr[ent.label_]
                })
        
  
        for pattern_text, pattern_label in self.echr_pii_patterns:
            if pattern_text in text:
                detected.append({"text": pattern_text, "label": pattern_label})
        
        return detected
    
    
    def calculate_pdr(self, original_text, anonymized_text, entity_annotations):
        total_entities = len(entity_annotations)
        if total_entities == 0:
            return 1.0
        successfully_anonymized = 0
        for entity in entity_annotations:
            if entity['span_text'] in original_text and entity['span_text'] not in anonymized_text:
                successfully_anonymized += 1
        return successfully_anonymized / total_entities
    
    def calculate_sels(self, anonymized_text, sensitive_entities):
        total_sensitivity = len(sensitive_entities)
        if total_sensitivity == 0:
            return 0.0
        leaked_sensitivity = sum(1 for entity in sensitive_entities 
                               if entity['span_text'] in anonymized_text)
        return leaked_sensitivity / total_sensitivity
    
    
    def calculate_iar(self, anonymized_text, context_information):
        anonymized_embedding = self.sentence_model.encode(anonymized_text, convert_to_tensor=True)
        context_embedding = self.sentence_model.encode(context_information, convert_to_tensor=True)
        similarity = util.cos_sim(anonymized_embedding, context_embedding).item()
        return 1 - similarity
    
    def calculate_cpp(self, original_text, anonymized_text):
        original_embedding = self.sentence_model.encode(original_text, convert_to_tensor=True)
        anonymized_embedding = self.sentence_model.encode(anonymized_text, convert_to_tensor=True)
        return util.cos_sim(original_embedding, anonymized_embedding).item()