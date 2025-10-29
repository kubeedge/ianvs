"""
Saliency-Guided Masking for Privacy Protection

This module implements attention-based saliency masking to selectively suppress
important tokens while preserving semantic meaning for LLM inference.
"""

import logging
import numpy as np
import torch
from typing import Dict, List, Any, Tuple, Optional
from transformers import AutoTokenizer, AutoModel
import re

logger = logging.getLogger(__name__)


class SaliencyMasking:
    """
    Implements saliency-guided masking for privacy-preserving text processing.
    
    Uses attention mechanisms to identify important tokens and selectively
    mask them based on privacy requirements while preserving utility.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize saliency masking module.
        
        Args:
            config: Configuration dictionary containing masking parameters
        """
        self.config = config
        self.anonymization_config = config.get('anonymization', {})
        
        # Masking parameters
        self.general_mask_ratio = self.anonymization_config.get('general_mask_ratio', 0.4)
        self.high_sensitivity_mask_ratio = self.anonymization_config.get('high_sensitivity_mask_ratio', 0.6)
        self.saliency_threshold = self.anonymization_config.get('saliency_threshold', 0.3)
        
        # Special tokens and patterns to preserve
        self.preserve_patterns = [
            r'\[SEP\]', r'\[CLS\]', r'\[PAD\]', r'\[UNK\]', r'\[MASK\]',
            r'<\w+>', r'</\w+>',  # HTML-like tags
            r'\d+',  # Numbers might be important for semantic preservation
        ]
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize tokenizer for attention analysis
        self._init_tokenizer()
        
        logger.info("Saliency Masking module initialized")
    
    def _init_tokenizer(self):
        """Initialize tokenizer for attention analysis."""
        try:
            # Use a lightweight model for attention analysis
            model_name = "bert-base-chinese"  # Good for Chinese text analysis
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.attention_model = AutoModel.from_pretrained(model_name)
            self.attention_model.to(self.device)
            self.attention_model.eval()
            
            logger.info(f"Attention model loaded: {model_name}")
            
        except Exception as e:
            logger.warning(f"Failed to load attention model: {e}")
            self.tokenizer = None
            self.attention_model = None
    
    def apply_masking(self, text: str, mask_ratio: float, **kwargs) -> Dict[str, Any]:
        """
        Apply saliency-guided masking to text.
        
        Args:
            text: Input text to mask
            mask_ratio: Fraction of tokens to mask (0.0 to 1.0)
            **kwargs: Additional parameters
            
        Returns:
            dict: Masked text with metadata
        """
        if not isinstance(text, str) or not text.strip():
            return {
                'masked_text': text,
                'original_text': text,
                'mask_ratio_applied': 0.0,
                'tokens_masked': 0,
                'saliency_scores': []
            }
        
        # Tokenize text
        tokens = self._tokenize_text(text)
        
        # Calculate saliency scores
        saliency_scores = self._calculate_saliency_scores(text, tokens)
        
        # Determine tokens to mask
        mask_indices = self._select_tokens_to_mask(tokens, saliency_scores, mask_ratio)
        
        # Apply masking
        masked_tokens = self._apply_token_masking(tokens, mask_indices)
        
        # Reconstruct text
        masked_text = self._reconstruct_text(masked_tokens, tokens)
        
        return {
            'masked_text': masked_text,
            'original_text': text,
            'mask_ratio_applied': len(mask_indices) / len(tokens) if tokens else 0.0,
            'tokens_masked': len(mask_indices),
            'total_tokens': len(tokens),
            'saliency_scores': saliency_scores,
            'mask_indices': mask_indices,
            'masking_strategy': kwargs.get('strategy', 'saliency_guided')
        }
    
    def _tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text for masking analysis.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        if self.tokenizer:
            # Use transformer tokenizer
            tokens = self.tokenizer.tokenize(text)
        else:
            # Fallback to simple whitespace tokenization
            tokens = text.split()
        
        return tokens
    
    def _calculate_saliency_scores(self, text: str, tokens: List[str]) -> List[float]:
        """
        Calculate saliency scores for each token.
        
        Args:
            text: Original text
            tokens: Tokenized text
            
        Returns:
            List of saliency scores
        """
        if not self.attention_model or not tokens:
            # Fallback to simple heuristic scoring
            return self._heuristic_saliency_scores(tokens)
        
        try:
            # Use attention weights from transformer model
            return self._attention_based_saliency(text, tokens)
        except Exception as e:
            logger.warning(f"Attention-based saliency failed: {e}")
            return self._heuristic_saliency_scores(tokens)
    
    def _attention_based_saliency(self, text: str, tokens: List[str]) -> List[float]:
        """
        Calculate saliency scores using transformer attention weights.
        
        Args:
            text: Original text
            tokens: Tokenized text
            
        Returns:
            List of saliency scores
        """
        # Encode text
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, 
                               max_length=512, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.attention_model(**inputs, output_attentions=True)
            attentions = outputs.attentions  # List of attention matrices
        
        # Average attention across layers and heads
        # Shape: (layers, batch, heads, seq_len, seq_len)
        attention_stack = torch.stack(attentions)  # (layers, batch, heads, seq_len, seq_len)
        averaged_attention = attention_stack.mean(dim=(0, 2))  # Average over layers and heads
        
        # Extract attention scores for each token
        # Use attention from [CLS] token to all other tokens as importance measure
        cls_attention = averaged_attention[0, 0, 1:]  # Skip [CLS] token itself
        
        # Convert to numpy and normalize
        saliency_scores = cls_attention.cpu().numpy()
        saliency_scores = (saliency_scores - saliency_scores.min()) / (saliency_scores.max() - saliency_scores.min() + 1e-8)
        
        # Map back to original tokens (handle subword tokenization)
        token_scores = self._map_subword_to_word_scores(saliency_scores, tokens, inputs['input_ids'][0])
        
        return token_scores.tolist()
    
    def _heuristic_saliency_scores(self, tokens: List[str]) -> List[float]:
        """
        Calculate saliency scores using heuristic methods.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of saliency scores
        """
        scores = []
        
        for token in tokens:
            score = 0.0
            
            # Length-based scoring (longer tokens often more important)
            score += min(1.0, len(token) / 10.0) * 0.3
            
            # Content-based scoring
            if self._is_named_entity_like(token):
                score += 0.8  # High importance for named entities
            elif self._is_functional_word(token):
                score += 0.1  # Low importance for functional words
            elif self._contains_digits(token):
                score += 0.6  # Medium-high importance for numbers
            elif self._is_punctuation(token):
                score += 0.05  # Very low importance for punctuation
            else:
                score += 0.4  # Default importance for content words
            
            # Frequency-based adjustment (less frequent = more important)
            score += self._frequency_adjustment(token)
            
            scores.append(min(1.0, score))
        
        return scores
    
    def _map_subword_to_word_scores(self, subword_scores: np.ndarray, 
                                   original_tokens: List[str], 
                                   input_ids: torch.Tensor) -> np.ndarray:
        """
        Map subword attention scores back to original word tokens.
        
        Args:
            subword_scores: Attention scores for subword tokens
            original_tokens: Original word tokens
            input_ids: Input token IDs from tokenizer
            
        Returns:
            Mapped scores for original tokens
        """
        # This is a simplified mapping - in practice, you'd need more sophisticated alignment
        # For now, we'll just truncate or pad to match original token length
        
        if len(subword_scores) >= len(original_tokens):
            return subword_scores[:len(original_tokens)]
        else:
            # Pad with average score
            avg_score = subword_scores.mean() if len(subword_scores) > 0 else 0.5
            padded_scores = np.concatenate([
                subword_scores, 
                np.full(len(original_tokens) - len(subword_scores), avg_score)
            ])
            return padded_scores
    
    def _select_tokens_to_mask(self, tokens: List[str], saliency_scores: List[float], 
                              mask_ratio: float) -> List[int]:
        """
        Select tokens to mask based on saliency scores and privacy requirements.
        
        Args:
            tokens: List of tokens
            saliency_scores: Saliency scores for each token
            mask_ratio: Target masking ratio
            
        Returns:
            List of token indices to mask
        """
        if not tokens or mask_ratio <= 0:
            return []
        
        num_tokens_to_mask = int(len(tokens) * mask_ratio)
        
        # Create list of (index, score, token) tuples
        token_info = [(i, score, token) for i, (score, token) in enumerate(zip(saliency_scores, tokens))]
        
        # Filter out tokens that should be preserved
        maskable_tokens = [(i, score, token) for i, score, token in token_info 
                          if not self._should_preserve_token(token)]
        
        if not maskable_tokens:
            return []
        
        # Sort by saliency score (descending - mask most salient tokens first for privacy)
        maskable_tokens.sort(key=lambda x: x[1], reverse=True)
        
        # Select top tokens to mask, but consider diversity
        selected_indices = []
        
        # Primary selection based on saliency
        primary_count = min(num_tokens_to_mask, len(maskable_tokens))
        selected_indices.extend([idx for idx, _, _ in maskable_tokens[:primary_count]])
        
        # If we need more tokens, add random selection for diversity
        remaining_needed = num_tokens_to_mask - len(selected_indices)
        if remaining_needed > 0:
            remaining_tokens = [idx for idx, _, _ in maskable_tokens[primary_count:]]
            if remaining_tokens:
                np.random.shuffle(remaining_tokens)
                selected_indices.extend(remaining_tokens[:remaining_needed])
        
        return sorted(selected_indices)
    
    def _apply_token_masking(self, tokens: List[str], mask_indices: List[int]) -> List[str]:
        """
        Apply masking to selected tokens.
        
        Args:
            tokens: Original tokens
            mask_indices: Indices of tokens to mask
            
        Returns:
            List of masked tokens
        """
        masked_tokens = tokens.copy()
        
        for idx in mask_indices:
            if idx < len(masked_tokens):
                original_token = masked_tokens[idx]
                masked_tokens[idx] = self._generate_mask_token(original_token)
        
        return masked_tokens
    
    def _generate_mask_token(self, original_token: str) -> str:
        """
        Generate appropriate mask token for the original token.
        
        Args:
            original_token: Original token to mask
            
        Returns:
            Masked token
        """
        # Different masking strategies based on token type
        if self._is_named_entity_like(original_token):
            return "[NAME]"
        elif self._contains_digits(original_token):
            return "[NUM]"
        elif self._is_punctuation(original_token):
            return original_token  # Keep punctuation for structure
        elif len(original_token) <= 2:
            return "[X]"
        else:
            # Replace with generic mask token
            return "[MASK]"
    
    def _reconstruct_text(self, masked_tokens: List[str], original_tokens: List[str]) -> str:
        """
        Reconstruct text from masked tokens.
        
        Args:
            masked_tokens: Tokens after masking
            original_tokens: Original tokens for reference
            
        Returns:
            Reconstructed text
        """
        if self.tokenizer:
            # Use tokenizer's conversion if available
            try:
                return self.tokenizer.convert_tokens_to_string(masked_tokens)
            except Exception as e:
                logger.warning(f"Failed to reconstruct text with tokenizer, falling back to joining: {e}")
        
        # Fallback to simple joining
        return " ".join(masked_tokens)
    
    def _should_preserve_token(self, token: str) -> bool:
        """Check if token should be preserved from masking."""
        # Check against preserve patterns
        for pattern in self.preserve_patterns:
            if re.match(pattern, token):
                return True
        
        # Preserve very short functional tokens
        if len(token) <= 1 and token in ['a', 'I', '的', '是', '在', '了']:
            return True
        
        return False
    
    def _is_named_entity_like(self, token: str) -> bool:
        """Check if token looks like a named entity."""
        # Chinese names (simplified check)
        if re.match(r'^[\u4e00-\u9fff]{2,4}$', token):
            return True
        
        # English capitalized words
        if token[0].isupper() and len(token) > 1:
            return True
        
        return False
    
    def _is_functional_word(self, token: str) -> bool:
        """Check if token is a functional word."""
        functional_words = {
            # English
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            # Chinese
            '的', '了', '在', '是', '我', '你', '他', '她', '它', '们', '这', '那',
            '有', '没', '不', '很', '也', '都', '还', '就', '会', '能', '要', '可以'
        }
        
        return token.lower() in functional_words
    
    def _contains_digits(self, token: str) -> bool:
        """Check if token contains digits."""
        return any(char.isdigit() for char in token)
    
    def _is_punctuation(self, token: str) -> bool:
        """Check if token is punctuation."""
        return len(token) == 1 and not token.isalnum()
    
    def _frequency_adjustment(self, token: str) -> float:
        """Adjust score based on estimated token frequency."""
        # This is a simplified frequency estimation
        # In practice, you'd use a real frequency dictionary
        
        if len(token) == 1:
            return -0.2  # Very common single characters
        elif len(token) <= 3:
            return -0.1  # Common short words
        elif len(token) >= 8:
            return 0.3   # Uncommon long words
        else:
            return 0.0   # Neutral
    
    def get_masking_statistics(self, masking_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate statistics about the masking operation.
        
        Args:
            masking_result: Result from apply_masking
            
        Returns:
            Dictionary with masking statistics
        """
        total_tokens = masking_result.get('total_tokens', 0)
        tokens_masked = masking_result.get('tokens_masked', 0)
        saliency_scores = masking_result.get('saliency_scores', [])
        
        if not saliency_scores:
            return {'error': 'No saliency scores available'}
        
        stats = {
            'masking_ratio': {
                'target': masking_result.get('mask_ratio_applied', 0.0),
                'achieved': tokens_masked / total_tokens if total_tokens > 0 else 0.0
            },
            'saliency_analysis': {
                'min_score': min(saliency_scores),
                'max_score': max(saliency_scores),
                'mean_score': sum(saliency_scores) / len(saliency_scores),
                'std_score': np.std(saliency_scores)
            },
            'token_analysis': {
                'total_tokens': total_tokens,
                'tokens_masked': tokens_masked,
                'tokens_preserved': total_tokens - tokens_masked
            },
            'privacy_impact': {
                'information_removed': tokens_masked / total_tokens if total_tokens > 0 else 0.0,
                'structure_preserved': True,  # We preserve punctuation and structure
                'semantic_impact': 'medium' if tokens_masked / total_tokens > 0.5 else 'low'
            }
        }
        
        return stats

