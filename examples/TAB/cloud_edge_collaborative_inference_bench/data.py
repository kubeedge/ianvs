import json
import os
import numpy as np
from typing import List, Dict
from sedna.datasources import BaseDataSource
from sedna.common.class_factory import ClassFactory, ClassType
import logging

logging.basicConfig(level=logging.INFO)

@ClassFactory.register(ClassType.GENERAL, alias="ECHRDataProcessor")
class ECHRDataProcessor:
    
    def __init__(self, cache_dir="./cache/echr_dataset", **kwargs):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.processed_data = None  
    
    def __call__(self, dataset: BaseDataSource) -> BaseDataSource:
        """Transform the dataset for ECHR data processing"""
        try:
            def parse_question_embedded(q: str):
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
           
            processed_items = []
            
            x_data = dataset.x
            if isinstance(x_data, np.ndarray):
                x_data = x_data.tolist()  
            
           
            y_data = dataset.y
            if isinstance(y_data, np.ndarray):
                y_data = y_data.tolist()  
            
            for x, y in zip(x_data, y_data):
               
                if isinstance(x, dict):
                    if "question" in x:  
                        text_val, ann = parse_question_embedded(x.get("question", ""))
                        doc = {
                            "text": text_val,
                            "annotations": ann or x.get("annotations", {}),
                            "doc_id": x.get("doc_id")
                        }
                    else:
                        doc = {
                            "text": x.get("text", ""),
                            "annotations": x.get("annotations", {}),
                            "doc_id": x.get("doc_id")
                        }
                else:
                    text = str(x) if not isinstance(x, str) else x
                    text_val, ann = parse_question_embedded(text)
                    if not ann:
                        if not isinstance(y, dict):
                            try:
                                y = json.loads(y) if isinstance(y, str) else {}
                            except:
                                y = {}
                        ann = y.get("annotations", {})
                    doc = {
                        "text": text_val,
                        "annotations": ann,
                        "doc_id": (y.get("doc_id") if isinstance(y, dict) else None)
                    }
                
                sensitive_entities = []
                total_mentions = 0
                id_type_stats = {}
                for annotator in doc.get("annotations", {}).values():
                    for entity in annotator.get("entity_mentions", []):
                        total_mentions += 1
                        id_type = entity.get("identifier_type")
                        if id_type:
                            id_type_stats[id_type] = id_type_stats.get(id_type, 0) + 1
                        if entity.get("identifier_type") in ["DIRECT", "QUASI"]:
                            sensitive_entities.append({
                                "span_text": entity["span_text"],
                                "entity_type": entity["entity_type"],
                                "start_offset": entity["start_offset"],
                                "end_offset": entity["end_offset"],
                                "sensitivity": 5 if entity["identifier_type"] == "DIRECT" else 3,
                                "identifier_type": entity["identifier_type"],
                                "entity_id": entity["entity_id"]
                            })
                
                processed_item = {
                    "text": doc["text"],
                    "doc_id": doc.get("doc_id"),
                    "sensitive_entities": sensitive_entities,
                    "raw_doc": doc
                }
                logging.info(
                    f"Processed doc_id {doc.get('doc_id')}: "
                    f"mentions_total={total_mentions}, id_type_stats={id_type_stats}, "
                    f"selected_sensitive={len(sensitive_entities)}"
                )
               

                processed_items.append(processed_item)
            
            dataset.x = processed_items
            self.processed_data = processed_items
        except Exception as e:
            raise RuntimeError(f"Failed to transform dataset for ECHR Data Processor: {e}") from e
        
        return dataset
    
    def process(self, dataset: BaseDataSource) -> List[Dict]:
       
        self.processed_data = []
        
        x_data = dataset.x
        if isinstance(x_data, np.ndarray):
            x_data = x_data.tolist()
        
        y_data = dataset.y
        if isinstance(y_data, np.ndarray):
            y_data = y_data.tolist()
        
        for x, y in zip(x_data, y_data):
            if isinstance(x, dict):
                if "question" in x:
                    def parse_question_embedded_local(q: str):
                        marker = "\n\nANNOTATIONS_JSON="
                        if isinstance(q, str) and marker in q:
                            head, tail = q.rsplit(marker, 1)
                            try:
                                return head.strip(), json.loads(tail.strip())
                            except Exception:
                                return head.strip(), {}
                        if isinstance(q, str) and ("<ANN>" in q and "</ANN>" in q):
                            try:
                                head, rest = q.split("<ANN>", 1)
                                json_str, _ = rest.split("</ANN>", 1)
                                return head.strip(), json.loads(json_str.strip())
                            except Exception:
                                return q, {}
                        return q, {}
                    text_val, ann = parse_question_embedded_local(x.get("question", ""))
                    doc = {
                        "text": text_val,
                        "annotations": ann or x.get("annotations", {}),
                        "doc_id": x.get("doc_id")
                    }
                else:
                    doc = {
                        "text": x.get("text", ""),
                        "annotations": x.get("annotations", {}),
                        "doc_id": x.get("doc_id")
                    }
            else:
                text = str(x) if not isinstance(x, str) else x
                text_val, ann = text, {}
                marker = "\n\nANNOTATIONS_JSON="
                if isinstance(text, str) and marker in text:
                    head, tail = text.rsplit(marker, 1)
                    text_val = head.strip()
                    try:
                        ann = json.loads(tail.strip())
                    except Exception:
                        ann = {}
                elif isinstance(text, str) and ("<ANN>" in text and "</ANN>" in text):
                    try:
                        head, rest = text.split("<ANN>", 1)
                        json_str, _ = rest.split("</ANN>", 1)
                        text_val = head.strip()
                        ann = json.loads(json_str.strip())
                    except Exception:
                        ann = {}
                if not ann:
                    
                    if not isinstance(y, dict):
                        try:
                            y = json.loads(y) if isinstance(y, str) else {}
                        except:
                            y = {}
                    ann = y.get("annotations", {})
                doc = {
                    "text": text_val,
                    "annotations": ann,
                    "doc_id": (y.get("doc_id") if isinstance(y, dict) else None)
                }
            
            sensitive_entities = []
            total_mentions = 0
            id_type_stats = {}
            for annotator in doc.get("annotations", {}).values():
                for entity in annotator.get("entity_mentions", []):
                    total_mentions += 1
                    id_type = entity.get("identifier_type")
                    if id_type:
                        id_type_stats[id_type] = id_type_stats.get(id_type, 0) + 1
                    if entity.get("identifier_type") in ["DIRECT", "QUASI"]:
                        sensitive_entities.append({
                            "span_text": entity["span_text"],
                            "entity_type": entity["entity_type"],
                            "start_offset": entity["start_offset"],
                            "end_offset": entity["end_offset"],
                            "sensitivity": 5 if entity["identifier_type"] == "DIRECT" else 3,
                            "identifier_type": entity["identifier_type"],
                            "entity_id": entity["entity_id"]
                        })
            
            processed_item = {
                "text": doc["text"],
                "doc_id": doc.get("doc_id"),
                "sensitive_entities": sensitive_entities,
                "raw_doc": doc
            }
            logging.info(
                f"Processed(doc mode=process) doc_id {doc.get('doc_id')}: "
                f"mentions_total={total_mentions}, id_type_stats={id_type_stats}, "
                f"selected_sensitive={len(sensitive_entities)}"
            )
            print(f"processed_item: {processed_item}")
            self.processed_data.append(processed_item)
        
        return self.processed_data
    
    def get_processed_data(self) -> List[Dict]:
        if self.processed_data is None:
            raise ValueError("Please call process() to process data first")
        return self.processed_data