<<<<<<< HEAD

import time
from sedna.common.class_factory import ClassFactory, ClassType
from sentence_transformers import SentenceTransformer, util
import logging
import json
import os


def _extract_annotations_from_text(text: str):
    if not isinstance(text, str):
        return {}
    marker = "\n\nANNOTATIONS_JSON="
    if marker in text:
        try:
            _, tail = text.rsplit(marker, 1)
            return json.loads(tail.strip())
        except Exception:
            return {}
    if "<ANN>" in text and "</ANN>" in text:
        try:
            _, rest = text.split("<ANN>", 1)
            json_str, _ = rest.split("</ANN>", 1)
            return json.loads(json_str.strip())
        except Exception:
            return {}
    return {}


def _normalize_result(paradigm_result):
    if isinstance(paradigm_result, list):
        if not paradigm_result:
            return {}
        return _normalize_result(paradigm_result[0])
    if isinstance(paradigm_result, tuple):
        if len(paradigm_result) >= 4 \
                and isinstance(paradigm_result[2], dict) \
                and isinstance(paradigm_result[3], dict):
            is_hard_example, res, edge_result, cloud_result = (
                paradigm_result[0], paradigm_result[1], paradigm_result[2], paradigm_result[3]
            )
            return {
                'is_hard_example': bool(is_hard_example),
                'result': res,
                'edge_result': edge_result,
                'cloud_result': cloud_result,
            }
        if len(paradigm_result) == 2 and isinstance(paradigm_result[1], dict):
            return paradigm_result[1]
        return {}
    if isinstance(paradigm_result, dict):
        return paradigm_result
    return {}


#Delay overhead (total duration of privacy processing and cloud inference, in seconds)
@ClassFactory.register(ClassType.GENERAL, alias="LatencyOverhead")
def LatencyOverhead(_, paradigm_result):
    pr = _normalize_result(paradigm_result)
    if not isinstance(pr, dict):
        return 0.0
    source = pr.get("inference_source")
    if source == "cloud" or (not source and pr.get("cloud_result")):
        return float((pr.get("cloud_result") or {}).get("total_time") or 0.0)
    return float((pr.get("edge_result") or {}).get("inference_time") or 0.0)


#Accuracy retention rate (semantic similarity between y and cloud results)
@ClassFactory.register(ClassType.GENERAL, alias="AccuracyPreservationRate")
def AccuracyPreservationRate(ground_truth_y, paradigm_result):
    pr = _normalize_result(paradigm_result)

    api_result = pr.get("result") if isinstance(pr, dict) else None
    generated_text = None
    if isinstance(api_result, dict):
        generated_text = api_result.get("generated_text")
    if not isinstance(generated_text, str) or not generated_text:
        cloud = pr.get('cloud_result') or {}
        if isinstance(cloud, dict):
            cloud_api_res = cloud.get('result') or {}
            if isinstance(cloud_api_res, dict):
                generated_text = cloud_api_res.get('generated_text')
    if not isinstance(generated_text, str) or not generated_text:
        return 0.0

    
    text = pr.get('original_text') or pr.get('text') or ''
    if not text:
        cloud = pr.get('cloud_result') or {}
        if isinstance(cloud, dict):
            text = cloud.get('original_text') or cloud.get('text') or ''

    sensitive_entities = pr.get('sensitive_entities')
    if not isinstance(sensitive_entities, list):
        cloud = pr.get('cloud_result') or {}
        if isinstance(cloud, dict):
            sensitive_entities = cloud.get('sensitive_entities')
    if not isinstance(sensitive_entities, list):
        sensitive_entities = []

    direct_entities = []
    for ent in sensitive_entities:
        if not isinstance(ent, dict):
            continue
        if ent.get("identifier_type") == "DIRECT":
            start = int(ent.get("start_offset", 0) or 0)
            end = int(ent.get("end_offset", 0) or 0)
            entity_type = ent.get("entity_type", "UNKNOWN")
            if isinstance(text, str) and 0 <= start < end <= len(text):
                direct_entities.append({
                    "start": start,
                    "end": end,
                    "entity_type": entity_type
                })


    direct_entities.sort(key=lambda x: x["end"], reverse=True)
    answer_text = text
    for ent in direct_entities:
        answer_text = answer_text[:ent["start"]] + ent["entity_type"] + answer_text[ent["end"]:]

    if not isinstance(answer_text, str) or not answer_text:
        if not isinstance(ground_truth_y, str) or not ground_truth_y:
            return 0.0
        answer_text = ground_truth_y

    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    local_st_model_dir = os.environ.get(
        "ST_MODEL_DIR",
        "./examples/TAB/cloud_edge_collaborative_inference_bench/all-MiniLM-L6-v2",
    )
    model = SentenceTransformer(local_st_model_dir)
    emb_y = model.encode(answer_text, convert_to_tensor=True)
    emb_pred = model.encode(generated_text, convert_to_tensor=True)
    return float(util.cos_sim(emb_y, emb_pred).item())

def measure_inference_time(model, data):
    start_time = time.time()
    model.inference(data)
    end_time = time.time()
    return end_time - start_time


=======
version https://git-lfs.github.com/spec/v1
oid sha256:44d198e8135fbe77c5df7c9f756f7f5131dee622366342493551a1b1eaa85f45
size 5326
>>>>>>> 9676c3e (ya toh aar ya toh par)
