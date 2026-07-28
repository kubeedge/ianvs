
import os
import sys
from sedna.common.class_factory import ClassFactory, ClassType
import logging
import json
try:
    from ..test_algorithms.privacy_desensitization.privacy_evaluator import PrivacyEvaluator
except (ImportError, ValueError):
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    bench_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    if bench_root not in sys.path:
        sys.path.insert(0, bench_root)
    from test_algorithms.privacy_desensitization.privacy_evaluator import PrivacyEvaluator



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
            head, rest = text.split("<ANN>", 1)
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


def _get_anonymized_text(paradigm_result: dict) -> str:
    txt = paradigm_result.get('protected_text') or paradigm_result.get('text')
    if isinstance(txt, str) and txt:
        return txt
    cloud = paradigm_result.get('cloud_result') or {}
    if isinstance(cloud, dict):
        txt = cloud.get('protected_text') or cloud.get('text')
        if isinstance(txt, str) and txt:
            return txt
        res = cloud.get('result')
        if isinstance(res, dict) and isinstance(res.get('generated_text'), str):
            return res['generated_text']
    edge = paradigm_result.get('edge_result') or {}
    if isinstance(edge, dict):
        res = edge.get('result')
        if isinstance(res, dict) and isinstance(res.get('generated_text'), str):
            return res['generated_text']
    return ''



@ClassFactory.register(ClassType.GENERAL, alias="PIIDetectionRate")
def PIIDetectionRate(_, paradigm_result):
    _log_paradigm_result(paradigm_result)
    evaluator = PrivacyEvaluator()
    paradigm_result = _normalize_result(paradigm_result)
    original_text = paradigm_result.get('original_text') or ''
    anonymized_text = _get_anonymized_text(paradigm_result)

    entity_list = []
    entity_list = paradigm_result.get('sensitive_entities')

    if not isinstance(entity_list, list):
        cloud = paradigm_result.get('cloud_result') or {}
        if isinstance(cloud, dict):
            entity_list = cloud.get('sensitive_entities')
    if not isinstance(entity_list, list):
        entity_list = []
        annotations = paradigm_result.get('annotations') or _extract_annotations_from_text(original_text)
        for annotator in (annotations or {}).values():
            for ent in annotator.get('entity_mentions', []):
                if 'span_text' in ent:
                    entity_list.append(ent)

    return evaluator.calculate_pdr(
        original_text,
        anonymized_text,
        entity_list
    )


@ClassFactory.register(ClassType.GENERAL, alias="SensitiveEntityLeakageScore")
def SensitiveEntityLeakageScore(_, paradigm_result):
    _log_paradigm_result(paradigm_result)
    paradigm_result = _normalize_result(paradigm_result)
    anonymized_text = _get_anonymized_text(paradigm_result)

    sensitive_entities = paradigm_result.get('sensitive_entities')
    if not isinstance(sensitive_entities, list):
        cloud = paradigm_result.get('cloud_result') or {}
        if isinstance(cloud, dict):
            sensitive_entities = cloud.get('sensitive_entities')
    if not isinstance(sensitive_entities, list):
        annotations = paradigm_result.get('annotations') or _extract_annotations_from_text(paradigm_result.get('original_text') or '')
        sensitive_entities = []
        for annotator in (annotations or {}).values():
            for ent in annotator.get('entity_mentions', []):
                if ent.get('identifier_type') in ("DIRECT", "QUASI"):
                    sensitive_entities.append(ent)

    weights = {"DIRECT": 5, "QUASI": 3, "NO_MASK": 0}
    total_weight = 0
    leaked_weight = 0
    for ent in sensitive_entities:
        id_type = ent.get('identifier_type', 'NO_MASK')
        weight = weights.get(id_type, 0)
        if id_type in ("DIRECT", "QUASI"):
            total_weight += weight
            span_text = ent.get('span_text', '')
            if span_text and span_text in anonymized_text:
                leaked_weight += weight

    if total_weight == 0:
        return 0.0
    return leaked_weight / total_weight


@ClassFactory.register(ClassType.GENERAL, alias="InferenceAttackResistance")
def InferenceAttackResistance(_, paradigm_result):
    _log_paradigm_result(paradigm_result)
    evaluator = PrivacyEvaluator()
    paradigm_result = _normalize_result(paradigm_result)
    anonymized_text = _get_anonymized_text(paradigm_result)

    api_result = paradigm_result.get('result') or {}
    aux_text = api_result.get('generated_text') if isinstance(api_result, dict) else None
    if not isinstance(aux_text, str) or not aux_text:
        try:
            import json
            aux_text = json.dumps(api_result, ensure_ascii=False)
        except Exception:
            aux_text = str(api_result)

    return evaluator.calculate_iar(
        anonymized_text,
        aux_text
    )


@ClassFactory.register(ClassType.GENERAL, alias="ContextualPrivacyPreservation")
def ContextualPrivacyPreservation(_, paradigm_result):
    _log_paradigm_result(paradigm_result)
    evaluator = PrivacyEvaluator()
    paradigm_result = _normalize_result(paradigm_result)
    original_text = paradigm_result.get('original_text') or ''
    anonymized_text = _get_anonymized_text(paradigm_result)
    return evaluator.calculate_cpp(
        original_text,
        anonymized_text
    )





def _log_paradigm_result(paradigm_result):
    try:
        logging.info("[metric dbg] paradigm_result type=%s", type(paradigm_result))
        if isinstance(paradigm_result, (list, tuple)):
            logging.info("[metric dbg] paradigm_result len=%d (showing first)", len(paradigm_result))
            if paradigm_result:
                pr0 = paradigm_result[0]
                logging.info("[metric dbg] paradigm_result[0] type=%s", type(pr0))
                if isinstance(pr0, dict):
                    logging.info("[metric dbg] paradigm_result[0] keys=%s", list(pr0.keys()))
        elif isinstance(paradigm_result, dict):
            logging.info("[metric dbg] paradigm_result keys=%s", list(paradigm_result.keys()))
            if 'cloud_result' in paradigm_result and isinstance(paradigm_result['cloud_result'], dict):
                logging.info("[metric dbg] cloud_result keys=%s", list(paradigm_result['cloud_result'].keys()))
            if 'edge_result' in paradigm_result and isinstance(paradigm_result['edge_result'], dict):
                logging.info("[metric dbg] edge_result keys=%s", list(paradigm_result['edge_result'].keys()))
    except Exception as e:
        logging.warning("[metric dbg] failed to log paradigm_result: %s", e)