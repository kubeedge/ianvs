from typing import Dict, Any
from ..test_algorithms.privacy_desensitization.privacy_evaluator import PrivacyEvaluator

def compute_pdr(tab_doc: Dict[str, Any], desensitized_text: str) -> float:
    return PrivacyEvaluator().calculate_pdr_on_tab(tab_doc, desensitized_text)

def compute_sels(tab_doc: Dict[str, Any], desensitized_text: str) -> float:
    return PrivacyEvaluator().calculate_sels_on_tab(tab_doc, desensitized_text)


