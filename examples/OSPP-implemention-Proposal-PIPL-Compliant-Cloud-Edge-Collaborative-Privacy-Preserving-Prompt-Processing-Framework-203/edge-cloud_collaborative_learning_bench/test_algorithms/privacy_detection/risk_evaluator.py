"""
Privacy Risk Evaluator

This module evaluates privacy risks in text data considering:
- PII entity density and sensitivity
- Context analysis and semantic understanding  
- PIPL compliance requirements
- Cross-border transmission risks
"""

import logging
import math
from typing import Dict, List, Any, Tuple
import numpy as np
from collections import Counter

logger = logging.getLogger(__name__)


class RiskEvaluator:
    """
    Evaluates privacy risks for text data to determine appropriate protection levels.
    
    Risk assessment considers:
    - PII entity types and density
    - Contextual sensitivity indicators
    - Regulatory compliance requirements
    - Cross-border transmission implications
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize risk evaluator with configuration.
        
        Args:
            config: Configuration dictionary containing risk weights and thresholds
        """
        self.config = config
        self.risk_weights = config.get('risk_weights', {})
        
        # Default risk weights if not provided
        self.default_weights = {
            'structured_pii': 0.8,
            'named_entities': 0.6,
            'semantic_context': 0.4,
            'entity_density': 0.5,
            'cross_border_risk': 0.7
        }
        
        # Entity type risk scores
        self.entity_risk_scores = {
            'ID_CARD': 1.0,
            'FINANCIAL': 0.95,
            'PHONE': 0.85,
            'EMAIL': 0.7,
            'PERSON': 0.6,
            'ADDRESS': 0.75,
            'ORG': 0.3,
            'LOCATION': 0.4,
            'DATE': 0.2,
            'TIME': 0.1
        }
        
        # Context risk indicators
        self.high_risk_contexts = [
            '医疗', '健康', '病历', '诊断', '治疗',  # Medical
            '金融', '银行', '贷款', '信贷', '投资',  # Financial
            '法律', '诉讼', '案件', '律师', '法院',  # Legal
            '政治', '政府', '官员', '党派', '选举',  # Political
            '宗教', '信仰', '教会', '寺庙', '神父',  # Religious
            'medical', 'health', 'diagnosis', 'treatment',
            'financial', 'bank', 'loan', 'credit', 'investment',
            'legal', 'lawsuit', 'court', 'lawyer', 'attorney',
            'political', 'government', 'official', 'party', 'election',
            'religious', 'belief', 'church', 'temple', 'priest'
        ]
        
        # Business contexts that may require protection
        self.business_risk_contexts = [
            '商业机密', '客户信息', '员工', '薪资', '合同',
            '供应商', '合作伙伴', '竞争对手', '市场策略',
            'business secret', 'customer info', 'employee', 'salary',
            'contract', 'supplier', 'partner', 'competitor', 'strategy'
        ]
        
        logger.info("Risk Evaluator initialized")
    
    def evaluate(self, text: str, entities: List[Dict[str, Any]]) -> float:
        """
        Evaluate overall privacy risk for text and detected entities.
        
        Args:
            text: Input text to evaluate
            entities: List of detected PII entities
            
        Returns:
            float: Risk score between 0.0 (low risk) and 1.0 (high risk)
        """
        if not text or not text.strip():
            return 0.0
        
        # Calculate component risk scores
        entity_risk = self._calculate_entity_risk(entities, len(text))
        context_risk = self._calculate_context_risk(text)
        density_risk = self._calculate_density_risk(entities, text)
        semantic_risk = self._calculate_semantic_risk(text)
        cross_border_risk = self._calculate_cross_border_risk(text, entities)
        
        # Get risk weights
        weights = self._get_risk_weights()
        
        # Calculate weighted risk score
        total_risk = (
            entity_risk * weights.get('structured_pii', 0.8) +
            context_risk * weights.get('semantic_context', 0.4) +
            density_risk * weights.get('entity_density', 0.5) +
            semantic_risk * weights.get('named_entities', 0.6) +
            cross_border_risk * weights.get('cross_border_risk', 0.7)
        )
        
        # Normalize by sum of weights
        total_weight = sum([
            weights.get('structured_pii', 0.8),
            weights.get('semantic_context', 0.4),
            weights.get('entity_density', 0.5),
            weights.get('named_entities', 0.6),
            weights.get('cross_border_risk', 0.7)
        ])
        
        normalized_risk = total_risk / total_weight if total_weight > 0 else 0.0
        
        # Apply risk amplification for high-sensitivity combinations
        amplified_risk = self._apply_risk_amplification(
            normalized_risk, entities, text
        )
        
        return min(1.0, max(0.0, amplified_risk))
    
    def _calculate_entity_risk(self, entities: List[Dict[str, Any]], text_length: int) -> float:
        """Calculate risk based on detected entities and their types."""
        if not entities:
            return 0.0
        
        total_risk = 0.0
        max_single_risk = 0.0
        
        for entity in entities:
            entity_type = entity.get('type', 'UNKNOWN')
            confidence = entity.get('confidence', 0.5)
            
            # Base risk score for entity type
            base_risk = self.entity_risk_scores.get(entity_type, 0.3)
            
            # Adjust by confidence
            adjusted_risk = base_risk * confidence
            
            # Adjust by entity-specific factors
            if entity.get('risk_level') == 'critical':
                adjusted_risk *= 1.2
            elif entity.get('risk_level') == 'high':
                adjusted_risk *= 1.1
            
            total_risk += adjusted_risk
            max_single_risk = max(max_single_risk, adjusted_risk)
        
        # Combine total and max risk (gives weight to both quantity and severity)
        entity_risk = 0.6 * min(1.0, total_risk / len(entities)) + 0.4 * max_single_risk
        
        return entity_risk
    
    def _calculate_context_risk(self, text: str) -> float:
        """Calculate risk based on contextual indicators."""
        text_lower = text.lower()
        
        # Check for high-risk contexts
        high_risk_matches = sum(1 for context in self.high_risk_contexts 
                               if context in text_lower)
        
        # Check for business risk contexts
        business_risk_matches = sum(1 for context in self.business_risk_contexts 
                                   if context in text_lower)
        
        # Calculate context risk
        high_risk_score = min(1.0, high_risk_matches * 0.3)
        business_risk_score = min(0.6, business_risk_matches * 0.2)
        
        # Additional indicators
        privacy_indicators = self._detect_privacy_indicators(text)
        urgency_indicators = self._detect_urgency_indicators(text)
        
        total_context_risk = (
            high_risk_score * 0.4 +
            business_risk_score * 0.3 +
            privacy_indicators * 0.2 +
            urgency_indicators * 0.1
        )
        
        return min(1.0, total_context_risk)
    
    def _calculate_density_risk(self, entities: List[Dict[str, Any]], text: str) -> float:
        """Calculate risk based on PII entity density."""
        if not entities or not text:
            return 0.0
        
        text_length = len(text)
        entity_count = len(entities)
        
        # Calculate entities per character
        density = entity_count / text_length
        
        # Calculate coverage (what percentage of text is PII)
        total_pii_chars = sum(entity.get('end', 0) - entity.get('start', 0) 
                             for entity in entities)
        coverage = total_pii_chars / text_length if text_length > 0 else 0
        
        # Normalize density (typical threshold: 1 entity per 100 characters)
        normalized_density = min(1.0, density * 100)
        
        # Combine density and coverage
        density_risk = 0.6 * normalized_density + 0.4 * coverage
        
        return density_risk
    
    def _calculate_semantic_risk(self, text: str) -> float:
        """Calculate risk based on semantic content analysis."""
        # Semantic patterns that indicate sensitive content
        sensitive_patterns = [
            '个人信息', '隐私', '保密', '机密', '敏感',
            '不公开', '限制访问', '内部使用', '仅限',
            'personal information', 'privacy', 'confidential',
            'sensitive', 'restricted', 'internal use', 'classified'
        ]
        
        relationship_patterns = [
            '家庭', '婚姻', '关系', '配偶', '子女', '父母',
            'family', 'marriage', 'relationship', 'spouse', 'children'
        ]
        
        financial_context = [
            '收入', '工资', '奖金', '债务', '资产', '财产',
            'income', 'salary', 'bonus', 'debt', 'assets', 'property'
        ]
        
        text_lower = text.lower()
        
        # Count semantic indicators
        sensitive_score = sum(1 for pattern in sensitive_patterns 
                             if pattern in text_lower)
        relationship_score = sum(1 for pattern in relationship_patterns 
                               if pattern in text_lower)
        financial_score = sum(1 for pattern in financial_context 
                             if pattern in text_lower)
        
        # Calculate semantic risk
        semantic_risk = (
            min(0.4, sensitive_score * 0.2) +
            min(0.3, relationship_score * 0.15) +
            min(0.3, financial_score * 0.15)
        )
        
        return semantic_risk
    
    def _calculate_cross_border_risk(self, text: str, entities: List[Dict[str, Any]]) -> float:
        """Calculate risk specific to cross-border data transmission."""
        # Factors that increase cross-border risk
        jurisdiction_indicators = [
            '中国', '美国', '欧盟', '境外', '跨境', '国际',
            'china', 'usa', 'eu', 'overseas', 'cross-border', 'international'
        ]
        
        regulatory_indicators = [
            'gdpr', 'pipl', '数据保护', '合规', '监管',
            'data protection', 'compliance', 'regulation'
        ]
        
        text_lower = text.lower()
        
        # Check for jurisdiction mentions
        jurisdiction_risk = min(0.4, sum(0.1 for indicator in jurisdiction_indicators 
                                        if indicator in text_lower))
        
        # Check for regulatory context
        regulatory_risk = min(0.3, sum(0.1 for indicator in regulatory_indicators 
                                      if indicator in text_lower))
        
        # Entity-based cross-border risk
        high_risk_entities = ['ID_CARD', 'FINANCIAL', 'PHONE']
        entity_risk = sum(0.1 for entity in entities 
                         if entity.get('type') in high_risk_entities)
        entity_cross_border_risk = min(0.3, entity_risk)
        
        total_cross_border_risk = jurisdiction_risk + regulatory_risk + entity_cross_border_risk
        
        return min(1.0, total_cross_border_risk)
    
    def _detect_privacy_indicators(self, text: str) -> float:
        """Detect explicit privacy-related language."""
        privacy_keywords = [
            '隐私政策', '数据保护', '个人数据', '用户信息',
            'privacy policy', 'data protection', 'personal data', 'user information'
        ]
        
        text_lower = text.lower()
        matches = sum(1 for keyword in privacy_keywords if keyword in text_lower)
        
        return min(1.0, matches * 0.25)
    
    def _detect_urgency_indicators(self, text: str) -> float:
        """Detect urgency indicators that might affect risk assessment."""
        urgency_keywords = [
            '紧急', '立即', '马上', '尽快', '优先',
            'urgent', 'immediate', 'asap', 'priority', 'critical'
        ]
        
        text_lower = text.lower()
        matches = sum(1 for keyword in urgency_keywords if keyword in text_lower)
        
        return min(1.0, matches * 0.2)
    
    def _apply_risk_amplification(self, base_risk: float, entities: List[Dict[str, Any]], text: str) -> float:
        """Apply risk amplification for high-sensitivity combinations."""
        amplification_factor = 1.0
        
        # Amplify risk if multiple high-risk entity types are present
        high_risk_types = set(entity.get('type') for entity in entities 
                             if entity.get('type') in ['ID_CARD', 'FINANCIAL', 'PHONE'])
        
        if len(high_risk_types) >= 2:
            amplification_factor *= 1.2
        
        if len(high_risk_types) >= 3:
            amplification_factor *= 1.3
        
        # Amplify risk for high-sensitivity contexts
        sensitive_contexts = self._count_sensitive_contexts(text)
        if sensitive_contexts >= 2:
            amplification_factor *= 1.15
        
        # Amplify risk for very high entity density
        entity_density = len(entities) / len(text) if text else 0
        if entity_density > 0.02:  # More than 2 entities per 100 characters
            amplification_factor *= 1.1
        
        return base_risk * amplification_factor
    
    def _count_sensitive_contexts(self, text: str) -> int:
        """Count number of different sensitive contexts present."""
        context_groups = [
            ['医疗', '健康', 'medical', 'health'],
            ['金融', '银行', 'financial', 'bank'],
            ['法律', '诉讼', 'legal', 'lawsuit'],
            ['政治', '政府', 'political', 'government'],
            ['宗教', '信仰', 'religious', 'belief']
        ]
        
        text_lower = text.lower()
        context_count = 0
        
        for group in context_groups:
            if any(keyword in text_lower for keyword in group):
                context_count += 1
        
        return context_count
    
    def _get_risk_weights(self) -> Dict[str, float]:
        """Get risk weights from configuration or use defaults."""
        weights = {}
        
        for key, default_value in self.default_weights.items():
            weights[key] = self.risk_weights.get(key, default_value)
        
        return weights
    
    def get_risk_breakdown(self, text: str, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get detailed breakdown of risk assessment.
        
        Args:
            text: Input text
            entities: Detected PII entities
            
        Returns:
            dict: Detailed risk analysis breakdown
        """
        entity_risk = self._calculate_entity_risk(entities, len(text))
        context_risk = self._calculate_context_risk(text)
        density_risk = self._calculate_density_risk(entities, text)
        semantic_risk = self._calculate_semantic_risk(text)
        cross_border_risk = self._calculate_cross_border_risk(text, entities)
        
        overall_risk = self.evaluate(text, entities)
        
        return {
            'overall_risk': overall_risk,
            'risk_level': self._categorize_risk_level(overall_risk),
            'component_risks': {
                'entity_risk': entity_risk,
                'context_risk': context_risk,
                'density_risk': density_risk,
                'semantic_risk': semantic_risk,
                'cross_border_risk': cross_border_risk
            },
            'risk_factors': {
                'high_risk_entities': len([e for e in entities 
                                         if e.get('type') in ['ID_CARD', 'FINANCIAL', 'PHONE']]),
                'total_entities': len(entities),
                'text_length': len(text),
                'entity_density': len(entities) / len(text) if text else 0,
                'sensitive_contexts': self._count_sensitive_contexts(text)
            },
            'recommendations': self._generate_recommendations(overall_risk, entities, text)
        }
    
    def _categorize_risk_level(self, risk_score: float) -> str:
        """Categorize numerical risk score into risk levels."""
        if risk_score >= 0.8:
            return 'critical'
        elif risk_score >= 0.6:
            return 'high'
        elif risk_score >= 0.4:
            return 'medium'
        elif risk_score >= 0.2:
            return 'low'
        else:
            return 'minimal'
    
    def _generate_recommendations(self, risk_score: float, entities: List[Dict[str, Any]], text: str) -> List[str]:
        """Generate privacy protection recommendations based on risk assessment."""
        recommendations = []
        
        if risk_score >= 0.8:
            recommendations.extend([
                "Apply maximum privacy protection measures",
                "Use high-sensitivity differential privacy parameters (ε ≤ 0.5)",
                "Consider blocking cross-border transmission",
                "Implement comprehensive audit logging"
            ])
        elif risk_score >= 0.6:
            recommendations.extend([
                "Apply strong privacy protection",
                "Use medium differential privacy parameters (ε ≤ 1.0)",
                "Require explicit consent for cross-border transmission",
                "Apply advanced anonymization techniques"
            ])
        elif risk_score >= 0.4:
            recommendations.extend([
                "Apply standard privacy protection",
                "Use standard differential privacy parameters (ε ≤ 1.5)",
                "Apply regular anonymization techniques"
            ])
        else:
            recommendations.extend([
                "Apply minimal privacy protection",
                "Standard anonymization may be sufficient"
            ])
        
        # Entity-specific recommendations
        high_risk_entities = [e for e in entities if e.get('type') in ['ID_CARD', 'FINANCIAL', 'PHONE']]
        if high_risk_entities:
            recommendations.append(f"Special attention required for {len(high_risk_entities)} high-risk entities")
        
        return recommendations
    
    def evaluate_risk(self, data: Dict[str, Any], context: str) -> Dict[str, Any]:
        """
        Evaluate privacy risk for given data and context.
        
        Args:
            data: Data dictionary containing type and value
            context: Context information for risk assessment
            
        Returns:
            Risk evaluation result with level, score, and recommendations
        """
        risk_result = {
            'risk_level': 'low',
            'risk_score': 0.0,
            'factors': [],
            'recommendations': [],
            'timestamp': None
        }
        
        try:
            # Extract data information
            data_type = data.get('type', 'unknown')
            data_value = data.get('value', '')
            
            # Initialize risk score
            base_risk_score = 0.0
            
            # Risk assessment based on data type
            if data_type in ['phone', 'id_card', 'bank_card', 'FINANCIAL']:
                base_risk_score = 0.8
                risk_result['risk_level'] = 'high'
                risk_result['factors'].append('Sensitive personal information')
                risk_result['recommendations'].append('Apply strong encryption')
                risk_result['recommendations'].append('Implement access controls')
            
            elif data_type in ['email', 'address', 'PERSON']:
                base_risk_score = 0.5
                risk_result['risk_level'] = 'medium'
                risk_result['factors'].append('Personal contact information')
                risk_result['recommendations'].append('Apply standard protection')
                risk_result['recommendations'].append('Consider anonymization')
            
            elif data_type in ['ORG', 'LOCATION']:
                base_risk_score = 0.3
                risk_result['risk_level'] = 'low'
                risk_result['factors'].append('Organizational or location information')
                risk_result['recommendations'].append('Apply basic protection')
            
            else:
                base_risk_score = 0.1
                risk_result['risk_level'] = 'very_low'
                risk_result['factors'].append('General information')
                risk_result['recommendations'].append('Minimal protection required')
            
            # Context-based risk adjustment
            context_lower = context.lower()
            
            # Cross-border transmission risk
            if 'cross_border' in context_lower or 'international' in context_lower:
                base_risk_score += 0.2
                risk_result['factors'].append('Cross-border transmission risk')
                risk_result['recommendations'].append('Use cross-border encryption')
                risk_result['recommendations'].append('Obtain explicit consent')
            
            # Medical or financial context
            if any(keyword in context_lower for keyword in ['medical', 'health', 'financial', 'banking']):
                base_risk_score += 0.15
                risk_result['factors'].append('Sensitive domain context')
                risk_result['recommendations'].append('Apply enhanced protection')
            
            # Legal or compliance context
            if any(keyword in context_lower for keyword in ['legal', 'compliance', 'audit', 'regulatory']):
                base_risk_score += 0.1
                risk_result['factors'].append('Legal compliance context')
                risk_result['recommendations'].append('Ensure regulatory compliance')
            
            # Data volume consideration
            if len(data_value) > 100:
                base_risk_score += 0.05
                risk_result['factors'].append('Large data volume')
                risk_result['recommendations'].append('Review data minimization')
            
            # Normalize risk score to [0, 1] range
            risk_score = min(base_risk_score, 1.0)
            risk_result['risk_score'] = risk_score
            
            # Adjust risk level based on final score
            if risk_score >= 0.8:
                risk_result['risk_level'] = 'critical'
            elif risk_score >= 0.6:
                risk_result['risk_level'] = 'high'
            elif risk_score >= 0.4:
                risk_result['risk_level'] = 'medium'
            elif risk_score >= 0.2:
                risk_result['risk_level'] = 'low'
            else:
                risk_result['risk_level'] = 'very_low'
            
            # Add timestamp
            from datetime import datetime
            risk_result['timestamp'] = datetime.now().isoformat()
            
            # Log risk evaluation
            logger.info(f"Risk evaluation completed: {data_type} -> {risk_result['risk_level']} (score: {risk_score:.2f})")
            
        except Exception as e:
            logger.error(f"Error in risk evaluation: {e}")
            risk_result['error'] = str(e)
            risk_result['risk_level'] = 'unknown'
            risk_result['risk_score'] = 0.0
        
        return risk_result

