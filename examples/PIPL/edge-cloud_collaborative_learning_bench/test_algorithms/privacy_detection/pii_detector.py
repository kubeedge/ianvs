<<<<<<< HEAD
"""
PII (Personally Identifiable Information) Detector

This module implements comprehensive PII detection using multiple approaches:
- Regex-based pattern matching
- Named Entity Recognition (NER)
- Context-aware semantic analysis
"""

import re
import logging
from typing import Dict, List, Any, Tuple
import spacy
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import jieba

# Import sedna ClassFactory for Ianvs integration
try:
    from sedna.common.class_factory import ClassFactory, ClassType
    SEDNA_AVAILABLE = True
except ImportError:
    SEDNA_AVAILABLE = False
    # Create dummy decorators if sedna is not available
    class ClassFactory:
        @staticmethod
        def register(class_type, alias=None):
            def decorator(cls):
                return cls
            return decorator
    class ClassType:
        GENERAL = "general"

logger = logging.getLogger(__name__)


@ClassFactory.register(ClassType.GENERAL, alias="PIIDetector")
class PIIDetector:
    """
    Comprehensive PII detection for Chinese and English text.
    
    Detects various types of personally identifiable information including:
    - Names, phone numbers, email addresses
    - Government IDs, addresses, financial information
    - Organization names and locations
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize PII detector with configuration.
        
        Args:
            config: Configuration dictionary containing detection parameters
        """
        self.config = config
        self.detection_methods = config.get('detection_methods', {})
        self.entity_types = config.get('detection_methods', {}).get('entity_types', [])
        
        # Initialize detection models
        self._init_regex_patterns()
        self._init_ner_models()
        
        logger.info("PII Detector initialized")
    
    def _init_regex_patterns(self):
        """Initialize regex patterns for PII detection."""
        
        # Chinese mobile phone patterns
        self.phone_patterns = [
            r'1[3-9]\d{9}',  # Chinese mobile phone number
            r'\+86[-\s]?1[3-9]\d{9}',  # Chinese mobile with country code
            r'(\d{3}[-\s]?)?\d{3}[-\s]?\d{4}',  # General phone format
            r'电话[:：]\s*(\+?86[-\s]?)?1[3-9]\d{9}',  # Phone with Chinese label
            r'手机[:：]\s*(\+?86[-\s]?)?1[3-9]\d{9}'   # Mobile with Chinese label
        ]
        
        # Email patterns
        self.email_patterns = [
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            r'邮箱[:：]\s*[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}',  # Email with Chinese label
            r'邮件[:：]\s*[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}'   # Email with Chinese label
        ]
        
        # Chinese ID card patterns
        self.id_card_patterns = [
            r'\b\d{17}[0-9Xx]\b',  # 18-digit Chinese ID card
            r'\b\d{15}\b',  # 15-digit Chinese ID card (old format)
            r'身份证号[:：]\s*\d{15,18}[0-9Xx]?',  # ID card with Chinese label
            r'证件号[:：]\s*\d{15,18}[0-9Xx]?'     # ID card with Chinese label
        ]
        
        # Address patterns
        self.address_patterns = [
            r'[北上广深]\w*[市区县]\w*[路街道]\w*号?\d*',  # Major Chinese cities
            r'\w+省\w*[市区县]\w*[路街道镇村]\w*',  # Provincial Chinese addresses
            r'地址[:：]\s*[\w\d\s-]+',  # Address with Chinese label
            r'住址[:：]\s*[\w\d\s-]+',  # Residence with Chinese label
            r'\d+\s+\w+\s+(Street|St|Avenue|Ave|Road|Rd)',  # English addresses
        ]
        
        # Name patterns (Chinese)
        self.name_patterns = [
            r'[姓名][:：]\s*[\u4e00-\u9fff]{2,4}',  # Name with Chinese label
            r'客户[:：]\s*[\u4e00-\u9fff]{2,4}',     # Customer with Chinese label
            r'联系人[:：]\s*[\u4e00-\u9fff]{2,4}',   # Contact with Chinese label
            r'[张王李赵刘陈杨黄周吴徐孙胡朱高林何郭马罗梁宋郑谢韩唐冯于董萧程曹袁邓许傅沈曾彭吕苏卢蒋蔡贾丁魏薛叶阎余潘杜戴夏钟汪田任姜范方石姚谭廖邹熊金陆郝孔白崔康毛邱秦江史顾侯邵孟龙万段漕钱汤尹黎易常武乔贺赖龚文][一-龯]{1,3}',  # Common Chinese surnames
            r'Mr\\.?\\s+[A-Z][a-z]+',  # English names
            r'Ms\\.?\\s+[A-Z][a-z]+',
            r'Dr\\.?\\s+[A-Z][a-z]+'
        ]
        
        # Financial information patterns
        self.financial_patterns = [
            r'\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}',  # Credit card number
            r'信用卡[:：]\s*\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}',  # Credit card with Chinese label
            r'银行卡[:：]\s*\d{16,19}',  # Bank card with Chinese label
            r'账号[:：]\s*\d{10,20}',    # Account with Chinese label
            r'IBAN[:：]?\s*[A-Z]{2}\d{2}[A-Z0-9]{4,30}'  # IBAN with Chinese label
        ]
        
        # Organization patterns
        self.organization_patterns = [
            r'[\u4e00-\u9fff]+公司',      # Chinese company names
            r'[\u4e00-\u9fff]+集团',      # Chinese group names
            r'[\u4e00-\u9fff]+企业',      # Chinese enterprise names
            r'[\u4e00-\u9fff]+有限责任公司',  # Chinese limited liability company
            r'[\u4e00-\u9fff]+股份有限公司',  # Chinese joint stock company
            r'[A-Z][a-z]+\\s+(Inc|Corp|LLC|Ltd|Company)',  # English company names
        ]
    
    def _init_ner_models(self):
        """Initialize Named Entity Recognition models."""
        try:
            # Load Chinese NER model
            ner_model_name = self.detection_methods.get('ner_model', 'hfl/chinese-bert-wwm-ext')
            
            self.ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_name)
            self.ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_name)
            self.ner_pipeline = pipeline(
                "ner",
                model=self.ner_model,
                tokenizer=self.ner_tokenizer,
                aggregation_strategy="simple"
            )
            
            logger.info(f"NER model loaded: {ner_model_name}")
            
        except Exception as e:
            logger.warning(f"Failed to load NER model: {e}")
            self.ner_pipeline = None
        
        # Initialize spaCy for additional NER
        try:
            self.nlp = spacy.load("zh_core_web_sm")
        except OSError:
            logger.warning("Chinese spaCy model not found")
            self.nlp = None
    
    def detect(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect PII entities in text using multiple methods.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of detected PII entities with metadata
        """
        if not isinstance(text, str) or not text.strip():
            return []
        
        detected_entities = []
        
        # Regex-based detection
        regex_entities = self._detect_with_regex(text)
        detected_entities.extend(regex_entities)
        
        # NER-based detection
        if self.ner_pipeline:
            ner_entities = self._detect_with_ner(text)
            detected_entities.extend(ner_entities)
        
        # spaCy-based detection
        if self.nlp:
            spacy_entities = self._detect_with_spacy(text)
            detected_entities.extend(spacy_entities)
        
        # Deduplicate and merge overlapping entities
        merged_entities = self._merge_entities(detected_entities)
        
        # Add confidence scores and risk levels
        enriched_entities = self._enrich_entities(merged_entities, text)
        
        return enriched_entities
    
    def _detect_with_regex(self, text: str) -> List[Dict[str, Any]]:
        """Detect PII using regex patterns."""
        entities = []
        
        pattern_groups = {
            'PHONE': self.phone_patterns,
            'EMAIL': self.email_patterns,
            'ID_CARD': self.id_card_patterns,
            'ADDRESS': self.address_patterns,
            'PERSON': self.name_patterns,
            'FINANCIAL': self.financial_patterns,
            'ORG': self.organization_patterns
        }
        
        for entity_type, patterns in pattern_groups.items():
            if entity_type in self.entity_types or not self.entity_types:
                for pattern in patterns:
                    matches = re.finditer(pattern, text, re.IGNORECASE)
                    for match in matches:
                        entities.append({
                            'type': entity_type,
                            'text': match.group(),
                            'start': match.start(),
                            'end': match.end(),
                            'method': 'regex',
                            'pattern': pattern,
                            'confidence': 0.9  # High confidence for regex matches
                        })
        
        return entities
    
    def _detect_with_ner(self, text: str) -> List[Dict[str, Any]]:
        """Detect PII using transformer-based NER."""
        if not self.ner_pipeline:
            return []
        
        entities = []
        
        try:
            # Run NER pipeline
            ner_results = self.ner_pipeline(text)
            
            for result in ner_results:
                entity_type = self._map_ner_label(result['entity_group'])
                
                if entity_type and (entity_type in self.entity_types or not self.entity_types):
                    entities.append({
                        'type': entity_type,
                        'text': result['word'],
                        'start': result['start'],
                        'end': result['end'],
                        'method': 'ner',
                        'confidence': result['score'],
                        'original_label': result['entity_group']
                    })
                    
        except Exception as e:
            logger.warning(f"NER detection failed: {e}")
        
        return entities
    
    def _detect_with_spacy(self, text: str) -> List[Dict[str, Any]]:
        """Detect PII using spaCy NER."""
        if not self.nlp:
            return []
        
        entities = []
        
        try:
            doc = self.nlp(text)
            
            for ent in doc.ents:
                entity_type = self._map_spacy_label(ent.label_)
                
                if entity_type and (entity_type in self.entity_types or not self.entity_types):
                    entities.append({
                        'type': entity_type,
                        'text': ent.text,
                        'start': ent.start_char,
                        'end': ent.end_char,
                        'method': 'spacy',
                        'confidence': 0.8,  # Default confidence for spaCy
                        'original_label': ent.label_
                    })
                    
        except Exception as e:
            logger.warning(f"spaCy detection failed: {e}")
        
        return entities
    
    def _map_ner_label(self, label: str) -> str:
        """Map NER model labels to standard PII types."""
        label_mapping = {
            'PER': 'PERSON',
            'PERSON': 'PERSON',
            'ORG': 'ORG',
            'ORGANIZATION': 'ORG',
            'LOC': 'LOCATION',
            'LOCATION': 'LOCATION',
            'GPE': 'LOCATION',  # Geopolitical entity
            'MISC': 'MISC'
        }
        
        return label_mapping.get(label.upper(), None)
    
    def _map_spacy_label(self, label: str) -> str:
        """Map spaCy labels to standard PII types."""
        label_mapping = {
            'PERSON': 'PERSON',
            'ORG': 'ORG',
            'GPE': 'LOCATION',
            'LOC': 'LOCATION',
            'FAC': 'LOCATION',  # Facility
            'MONEY': 'FINANCIAL',
            'DATE': 'DATE',
            'TIME': 'TIME'
        }
        
        return label_mapping.get(label, None)
    
    def _merge_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge overlapping entities and deduplicate."""
        if not entities:
            return []
        
        # Sort by start position
        sorted_entities = sorted(entities, key=lambda x: (x['start'], x['end']))
        merged = []
        
        current = sorted_entities[0]
        
        for next_entity in sorted_entities[1:]:
            # Check for overlap
            if (next_entity['start'] <= current['end'] and 
                next_entity['start'] >= current['start']):
                
                # Merge entities, keeping the one with higher confidence
                if next_entity['confidence'] > current['confidence']:
                    current = next_entity
                # If same confidence, prefer longer match
                elif (next_entity['confidence'] == current['confidence'] and
                      (next_entity['end'] - next_entity['start']) > (current['end'] - current['start'])):
                    current = next_entity
            else:
                merged.append(current)
                current = next_entity
        
        merged.append(current)
        return merged
    
    def _enrich_entities(self, entities: List[Dict[str, Any]], text: str) -> List[Dict[str, Any]]:
        """Add additional metadata to detected entities."""
        enriched = []
        
        for entity in entities:
            # Calculate risk level
            risk_level = self._calculate_risk_level(entity)
            
            # Add context
            context = self._extract_context(entity, text)
            
            # Enhance entity information
            enhanced_entity = {
                **entity,
                'risk_level': risk_level,
                'context': context,
                'sensitive': risk_level in ['high', 'critical'],
                'requires_protection': self._requires_protection(entity)
            }
            
            enriched.append(enhanced_entity)
        
        return enriched
    
    def _calculate_risk_level(self, entity: Dict[str, Any]) -> str:
        """Calculate risk level for detected entity."""
        entity_type = entity['type']
        confidence = entity['confidence']
        
        # Risk levels based on entity type
        high_risk_types = {'ID_CARD', 'FINANCIAL', 'PHONE'}
        medium_risk_types = {'EMAIL', 'PERSON', 'ADDRESS'}
        low_risk_types = {'ORG', 'LOCATION', 'DATE', 'TIME'}
        
        base_risk = 'low'
        if entity_type in high_risk_types:
            base_risk = 'high'
        elif entity_type in medium_risk_types:
            base_risk = 'medium'
        
        # Adjust based on confidence
        if confidence >= 0.9 and base_risk == 'high':
            return 'critical'
        elif confidence >= 0.8 and base_risk == 'high':
            return 'high'
        elif confidence >= 0.7 and base_risk == 'medium':
            return 'medium'
        elif confidence >= 0.6:
            return 'low'
        else:
            return 'very_low'
    
    def _extract_context(self, entity: Dict[str, Any], text: str, window: int = 50) -> str:
        """Extract context around detected entity."""
        start = max(0, entity['start'] - window)
        end = min(len(text), entity['end'] + window)
        
        context = text[start:end]
        
        # Mark the entity within context
        entity_start = entity['start'] - start
        entity_end = entity['end'] - start
        
        marked_context = (
            context[:entity_start] + 
            f"[{entity['type']}]" + 
            context[entity_start:entity_end] + 
            f"[/{entity['type']}]" + 
            context[entity_end:]
        )
        
        return marked_context.strip()
    
    def _requires_protection(self, entity: Dict[str, Any]) -> bool:
        """Determine if entity requires privacy protection."""
        high_protection_types = {'ID_CARD', 'FINANCIAL', 'PHONE', 'EMAIL'}
        
        # Calculate risk level if not already present
        if 'risk_level' not in entity:
            risk_level = self._calculate_risk_level(entity)
        else:
            risk_level = entity['risk_level']
        
        return (entity['type'] in high_protection_types or 
                risk_level in ['high', 'critical'])
    
    def get_entity_summary(self, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get summary statistics of detected entities."""
        if not entities:
            return {
                'total_entities': 0,
                'entity_types': {},
                'risk_distribution': {},
                'protection_required': 0
            }
        
        entity_types = {}
        risk_distribution = {}
        protection_required = 0
        
        for entity in entities:
            # Count entity types
            entity_type = entity['type']
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
            
            # Count risk levels
            risk_level = entity['risk_level']
            risk_distribution[risk_level] = risk_distribution.get(risk_level, 0) + 1
            
            # Count entities requiring protection
            if entity['requires_protection']:
                protection_required += 1
        
        return {
            'total_entities': len(entities),
            'entity_types': entity_types,
            'risk_distribution': risk_distribution,
            'protection_required': protection_required,
            'protection_ratio': protection_required / len(entities) if entities else 0
        }

=======
version https://git-lfs.github.com/spec/v1
oid sha256:cf6a0b388f4bedc308ef06fd8ae60dd6fade9cd00f2c4c2233480599d0f6e63f
size 17930
>>>>>>> 9676c3e (ya toh aar ya toh par)
