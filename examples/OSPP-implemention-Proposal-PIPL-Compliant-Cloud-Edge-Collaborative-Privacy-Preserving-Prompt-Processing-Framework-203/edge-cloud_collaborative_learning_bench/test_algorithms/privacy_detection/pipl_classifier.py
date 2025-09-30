"""
PIPL Compliance Classifier

This module implements privacy level classification according to PIPL (Personal Information Protection Law)
regulations, determining whether data contains sensitive information requiring special protection.
"""

import re
import logging
from typing import Dict, Any, List
import spacy
import jieba

logger = logging.getLogger(__name__)


class PIPLClassifier:
    """
    Classifies text content according to PIPL privacy sensitivity levels.
    
    Levels:
    - general: General content without sensitive personal information
    - high_sensitivity: Content containing sensitive personal information requiring special protection
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize PIPL classifier with configuration.
        
        Args:
            config: Configuration dictionary containing detection parameters
        """
        self.config = config
        self.cross_border_threshold = config.get('pipl_compliance', {}).get('cross_border_threshold', 0.5)
        self.high_sensitivity_threshold = config.get('pipl_compliance', {}).get('high_sensitivity_threshold', 0.7)
        
        # Initialize NLP models
        self._init_nlp_models()
        
        # PIPL sensitive patterns (Article 28 - Sensitive Personal Information)
        self._init_sensitive_patterns()
        
        logger.info("PIPL Classifier initialized")
    
    def _init_nlp_models(self):
        """Initialize NLP models for Chinese text processing."""
        try:
            # Load Chinese spaCy model if available
            self.nlp = spacy.load("zh_core_web_sm")
        except OSError:
            logger.warning("Chinese spaCy model not found, using basic tokenization")
            self.nlp = None
        
        # Initialize jieba for Chinese word segmentation
        jieba.initialize()
    
    def _init_sensitive_patterns(self):
        """Initialize patterns for detecting PIPL-defined sensitive information."""
        
        # PIPL Article 28: Sensitive Personal Information Categories
        self.sensitive_patterns = {
            # Biometric identifiers
            'biometric': [
                r'指纹', r'虹膜', r'声纹', r'掌纹', r'面部识别', r'生物特征',
                r'fingerprint', r'iris', r'voiceprint', r'facial recognition'
            ],
            
            # Religious beliefs  
            'religious': [
                r'宗教信仰', r'佛教', r'基督教', r'伊斯兰教', r'天主教', r'道教',
                r'religion', r'buddhist', r'christian', r'muslim', r'catholic'
            ],
            
            # Specific identity (race, ethnicity, political views)
            'identity': [
                r'种族', r'民族', r'政治观点', r'政治立场', r'党派', r'政治倾向',
                r'race', r'ethnicity', r'political view', r'political stance'
            ],
            
            # Trade union membership
            'union': [
                r'工会会员', r'工会', r'劳工组织', r'trade union', r'labor union'
            ],
            
            # Genetic/health information
            'health': [
                r'基因', r'遗传', r'健康状况', r'医疗记录', r'病历', r'诊断',
                r'genetic', r'health condition', r'medical record', r'diagnosis'
            ],
            
            # Sexual orientation
            'sexual': [
                r'性取向', r'性倾向', r'sexual orientation'
            ],
            
            # Location tracking
            'location': [
                r'实时位置', r'位置跟踪', r'GPS坐标', r'地理位置',
                r'location tracking', r'GPS coordinates', r'geolocation'
            ],
            
            # Children's information (under 14)
            'children': [
                r'14岁以下', r'儿童', r'未成年人', r'学生证', r'小学生',
                r'under 14', r'children', r'minor', r'student'
            ]
        }
        
        # Financial and identification patterns
        self.financial_patterns = [
            r'\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}',  # Credit card
            r'\d{6}[\s-]?\d{10}',  # Bank account
            r'银行账号', r'信用卡', r'支付密码', r'financial', r'bank account'
        ]
        
        # Personal identification patterns
        self.id_patterns = [
            r'\d{17}[\dXx]',  # Chinese ID card
            r'\d{15}',  # Old Chinese ID format
            r'护照号', r'passport', r'身份证', r'ID card'
        ]
    
    def classify(self, text: str) -> str:
        """
        Classify text according to PIPL privacy sensitivity levels.
        
        Args:
            text: Input text to classify
            
        Returns:
            str: Privacy level ('general' or 'high_sensitivity')
        """
        if not isinstance(text, str) or not text.strip():
            return 'general'
        
        # Calculate sensitivity score
        sensitivity_score = self._calculate_sensitivity_score(text)
        
        # Classify based on threshold
        if sensitivity_score >= self.high_sensitivity_threshold:
            return 'high_sensitivity'
        else:
            return 'general'
    
    def _calculate_sensitivity_score(self, text: str) -> float:
        """
        Calculate privacy sensitivity score based on PIPL criteria.
        
        Args:
            text: Input text
            
        Returns:
            float: Sensitivity score (0.0 to 1.0)
        """
        text_lower = text.lower()
        total_score = 0.0
        max_possible_score = 0.0
        
        # Check for PIPL-defined sensitive categories
        for category, patterns in self.sensitive_patterns.items():
            category_score = 0.0
            category_weight = self._get_category_weight(category)
            
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    category_score = 1.0
                    break
            
            total_score += category_score * category_weight
            max_possible_score += category_weight
        
        # Check for financial information
        financial_score = self._check_financial_patterns(text)
        total_score += financial_score * 0.8
        max_possible_score += 0.8
        
        # Check for identification information
        id_score = self._check_id_patterns(text)
        total_score += id_score * 0.9
        max_possible_score += 0.9
        
        # Normalize score
        if max_possible_score > 0:
            normalized_score = total_score / max_possible_score
        else:
            normalized_score = 0.0
        
        # Additional context analysis
        context_score = self._analyze_context_sensitivity(text)
        
        # Combine scores with weights
        final_score = 0.7 * normalized_score + 0.3 * context_score
        
        return min(1.0, final_score)
    
    def _get_category_weight(self, category: str) -> float:
        """Get weight for different PIPL sensitivity categories."""
        weights = {
            'biometric': 1.0,      # Highest sensitivity
            'health': 0.95,
            'children': 0.9,
            'identity': 0.85,
            'religious': 0.8,
            'sexual': 0.8,
            'location': 0.7,
            'union': 0.6
        }
        return weights.get(category, 0.5)
    
    def _check_financial_patterns(self, text: str) -> float:
        """Check for financial information patterns."""
        for pattern in self.financial_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return 1.0
        return 0.0
    
    def _check_id_patterns(self, text: str) -> float:
        """Check for identification information patterns."""
        for pattern in self.id_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return 1.0
        return 0.0
    
    def _analyze_context_sensitivity(self, text: str) -> float:
        """Analyze contextual indicators of sensitivity."""
        
        # Sensitive context keywords
        sensitive_contexts = [
            '隐私', '保密', '机密', '敏感', '私人', '个人信息',
            'privacy', 'confidential', 'sensitive', 'personal', 'private'
        ]
        
        # Business/legal contexts that may indicate sensitivity
        business_contexts = [
            '合同', '协议', '法律', '诉讼', '商业机密', '客户信息',
            'contract', 'agreement', 'legal', 'lawsuit', 'business secret'
        ]
        
        context_score = 0.0
        
        # Check for explicit privacy indicators
        for keyword in sensitive_contexts:
            if keyword in text.lower():
                context_score += 0.3
        
        # Check for business/legal contexts
        for keyword in business_contexts:
            if keyword in text.lower():
                context_score += 0.2
        
        # Length-based heuristic (longer texts may contain more sensitive info)
        if len(text) > 500:
            context_score += 0.1
        
        return min(1.0, context_score)
    
    def get_sensitivity_breakdown(self, text: str) -> Dict[str, Any]:
        """
        Get detailed breakdown of sensitivity analysis.
        
        Args:
            text: Input text to analyze
            
        Returns:
            dict: Detailed analysis breakdown
        """
        breakdown = {
            'privacy_level': self.classify(text),
            'overall_score': self._calculate_sensitivity_score(text),
            'category_scores': {},
            'detected_patterns': []
        }
        
        # Analyze each category
        for category, patterns in self.sensitive_patterns.items():
            category_detected = False
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    category_detected = True
                    breakdown['detected_patterns'].append({
                        'category': category,
                        'pattern': pattern,
                        'weight': self._get_category_weight(category)
                    })
                    break
            
            breakdown['category_scores'][category] = 1.0 if category_detected else 0.0
        
        # Check financial and ID patterns
        breakdown['financial_detected'] = self._check_financial_patterns(text) > 0
        breakdown['id_detected'] = self._check_id_patterns(text) > 0
        breakdown['context_score'] = self._analyze_context_sensitivity(text)
        
        return breakdown
    
    def is_cross_border_allowed(self, text: str) -> bool:
        """
        Determine if text can be transmitted across borders under PIPL.
        
        Args:
            text: Input text to check
            
        Returns:
            bool: True if cross-border transmission is allowed
        """
        privacy_level = self.classify(text)
        sensitivity_score = self._calculate_sensitivity_score(text)
        
        # High sensitivity data generally requires consent and additional protection
        if privacy_level == 'high_sensitivity':
            return False
        
        # Additional check based on sensitivity score threshold
        return sensitivity_score < self.cross_border_threshold

