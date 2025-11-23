<<<<<<< HEAD
# test_algorithms/hard_sample_mining/hard_sample_mining.py
"""
Full process inference scheduler (cloud/edge/privacy offloading), compatible with Sedna/Anvs model dependency automatic injection.
Implementation strategy:
1. YAML registration for "type: edgemodel/cloudmodel", the framework will be instantiated and automatically injected into the mining's __init__ kwargs.
2. Diversion/privacy judgment/accuracy detection/privacy cloud are all completed by this category.
"""
from sedna.common.class_factory import ClassFactory, ClassType
from core.common.log import LOGGER
import logging
from sentence_transformers import SentenceTransformer, util
import os

@ClassFactory.register(ClassType.HEM, alias="OracleRouter")
class OracleRouter:
    def __init__(self, **kwargs):
        self.edge_model = kwargs.get("edgemodel")
        self.cloud_model = kwargs.get("cloudmodel")
        self.edge_accuracy_threshold = kwargs.get("threshold", 1)
        self.fixed_privacy_method = kwargs.get("fixed_privacy_method", None)
        self.processed_data = kwargs.get("processed_data", None) 
        if not self.edge_model or not self.cloud_model:
            raise ValueError("Edge and cloud models must be injected for OracleRouter.")
    
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        local_st_model_dir = os.environ.get(
            "ST_MODEL_DIR",
            "./examples/TAB/cloud_edge_collaborative_inference_bench/all-MiniLM-L6-v2",
        )
        self._st_model = SentenceTransformer(local_st_model_dir)

    def try_attach_models(self, **kwargs):
        if not self.edge_model and 'edgemodel' in kwargs:
            self.edge_model = kwargs['edgemodel']
        if not self.cloud_model and 'cloudmodel' in kwargs:
            self.cloud_model = kwargs['cloudmodel']

    def __call__(self, data, **kwargs):
        self.try_attach_models(**kwargs)
        if not self.edge_model or not self.cloud_model:
            raise ValueError(
                f"Edge model: {self.edge_model}, Cloud model: {self.cloud_model} -- both models are required."
            )
        return self.determine_strategy(data)

    def has_privacy(self, data):
    
        text = data.get("text") or data.get("query")
        if not text:
            return False
        if data.get("sensitive_entities"):
            return True
        return False

    def determine_privacy_method(self, data):
        
        if self.fixed_privacy_method in ["regex", "ner", "dp"]:
            return self.fixed_privacy_method
        #Dynamic Maximum Sensitivity Rule
        sensitivities = [ent.get("sensitivity") for ent in data.get("sensitive_entities", []) if isinstance(ent.get("sensitivity"), (int, float))]
        if not sensitivities:
            return "regex"
        max_sen = max(sensitivities)
        if max_sen >= 4:
            return "regex"
        elif max_sen == 3:
            return "ner"
        else:
            return "dp"

    def _build_answer_text(self, text, sensitive_entities):
        if not isinstance(text, str):
            return ""
        direct_entities = []
        for ent in sensitive_entities or []:
            if not isinstance(ent, dict):
                continue
            if ent.get("identifier_type") == "DIRECT" or ent.get("dentifier_type") == "QUASI":
                start = int(ent.get("start_offset", 0) or 0)
                end = int(ent.get("end_offset", 0) or 0)
                entity_type = ent.get("entity_type", "UNKNOWN")
                if 0 <= start < end <= len(text):
                    direct_entities.append({
                        "start": start,
                        "end": end,
                        "entity_type": entity_type
                    })
        direct_entities.sort(key=lambda x: x["end"], reverse=True)
        answer_text = text
        for ent in direct_entities:
            answer_text = answer_text[:ent["start"]] + ent["entity_type"] + answer_text[ent["end"]:]
        return answer_text

    def _similarity(self, a, b):
        if not (isinstance(a, str) and isinstance(b, str) and a and b):
            return 0.0
        emb_a = self._st_model.encode(a, convert_to_tensor=True)
        emb_b = self._st_model.encode(b, convert_to_tensor=True)
        return float(util.cos_sim(emb_a, emb_b).item())

    def determine_strategy(self, data):
        """
        1. Conduct privacy check first
        2. Privacy free → Direct cloud inference
        3. Privacy included → Edge reasoning first
            3.1 Compliance → Return to Edge Results
            3.2 Failure to meet standards → Choose cloud based inference method
        """
        result = {
            "inference_source": None,
            "privacy_method": None,
            "edge_result": None,
            "cloud_result": None,
           
            "original_text": data.get("text"),
            "sensitive_entities": data.get("sensitive_entities", []),
            "raw_doc": data.get("raw_doc")
        }
        
       
        if not self.has_privacy(data):
            cr = self.cloud_model.inference({
                "text": data["text"],
                "sensitive_entities": data.get("sensitive_entities", [])
            }, privacy_method=None)
            result.update({"inference_source": "cloud", "cloud_result": cr})
            return result
      
        er = self.edge_model.infer({"query": data["text"]})
        edge_generated = None
        if isinstance(er, dict):
            edge_generated = (er.get("result") or {}).get("generated_text")
        
        answer_text = self._build_answer_text(data.get("text"), data.get("sensitive_entities", []))
        confidence = self._similarity(answer_text, edge_generated) if edge_generated else 0.0
        logging.info(f"[HEM] edge confidence(similarity to answer) = {confidence:.4f}")
        result.update({"edge_result": er})
        if confidence >= self.edge_accuracy_threshold:
            result.update({"inference_source": "edge"})
            return result
        
        pm = self.determine_privacy_method(data)
        logging.info(f"privacy_method: {pm}")
        cr = self.cloud_model.inference({
            "text": data["text"],
            "sensitive_entities": data.get("sensitive_entities", [])
        }, privacy_method=pm)
        result.update({"inference_source": "cloud", "privacy_method": pm, "cloud_result": cr})
        return result
=======
version https://git-lfs.github.com/spec/v1
oid sha256:f12c6ec43d3cee2a1b083150e8067bebba0f3332c668b0c9240f98a56a686d0b
size 6413
>>>>>>> 9676c3e (ya toh aar ya toh par)
