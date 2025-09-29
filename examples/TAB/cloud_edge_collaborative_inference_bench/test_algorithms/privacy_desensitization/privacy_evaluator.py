from typing import Dict, Any, List

class PrivacyEvaluator:
    def __init__(self, pii_detector=None, similarity_model=None):
        self.pii_detector = pii_detector
        self.similarity_model = similarity_model

    def calculate_pdr_on_tab(self, original_tab_doc: Dict[str, Any], desensitized_text: str) -> float:
        ground_truth_entities = original_tab_doc.get('entities', [])
        detected_pii = [] if self.pii_detector is None else self.pii_detector.detect(desensitized_text)

        masked_entities: List[Dict[str, Any]] = []
        for entity in ground_truth_entities:
            if entity.get('text') not in desensitized_text:
                masked_entities.append(entity)

        return (len(masked_entities) / len(ground_truth_entities)) if ground_truth_entities else 1.0

    def calculate_sels_on_tab(self, tab_doc: Dict[str, Any], desensitized_text: str) -> float:
        remaining_entities = []
        for entity in tab_doc.get('entities', []):
            if entity.get('text') in desensitized_text:
                remaining_entities.append(entity)

        if not remaining_entities:
            return 0.0

        weighted_leakage = sum(entity.get('sensitivity', 1) for entity in remaining_entities)
        max_possible_leakage = sum(entity.get('sensitivity', 1) for entity in tab_doc.get('entities', []))
        return (weighted_leakage / max_possible_leakage) if max_possible_leakage else 0.0


