<<<<<<< HEAD
"""
Differential Privacy Implementation

This module implements differential privacy mechanisms for protecting sensitive text data
while preserving utility for LLM inference. Includes:
- Gaussian noise addition with L2 norm clipping
- Privacy budget management and tracking
- Adaptive noise calibration based on sensitivity levels
"""

import logging
import math
import numpy as np
import torch
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class DifferentialPrivacy:
    """
    Differential Privacy mechanism for text data protection.
    
    Implements the Gaussian mechanism with L2 norm clipping and privacy budget tracking
    to ensure (ε, δ)-differential privacy for LLM inference tasks.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize differential privacy mechanism.
        
        Args:
            config: Configuration dictionary containing DP parameters
        """
        self.config = config
        self.dp_config = config.get('differential_privacy', {})
        self.budget_config = config.get('budget_management', {})
        
        # Privacy parameters
        self.general_params = self.dp_config.get('general', {})
        self.high_sensitivity_params = self.dp_config.get('high_sensitivity', {})
        
        # Budget management
        self.session_limit = self.budget_config.get('session_limit', 10.0)
        self.rate_limit = self.budget_config.get('rate_limit', 5)
        
        # Privacy budget tracking
        self.privacy_budget_consumed = 0.0
        self.query_count = 0
        self.session_start = datetime.now()
        self.query_history = []
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info("Differential Privacy module initialized")
    
    def add_noise(self, data: Any, dp_params: Dict[str, float]) -> Dict[str, Any]:
        """
        Add differential privacy noise to data.
        
        Args:
            data: Input data (text embeddings or vectors)
            dp_params: DP parameters (epsilon, delta, clipping_norm, noise_multiplier)
            
        Returns:
            dict: Noisy data with metadata
        """
        epsilon = dp_params.get('epsilon', 1.0)
        delta = dp_params.get('delta', 1e-5)
        clipping_norm = dp_params.get('clipping_norm', 1.0)
        noise_multiplier = dp_params.get('noise_multiplier', 1.1)
        
        # Check privacy budget
        if not self._check_privacy_budget(epsilon):
            raise ValueError(f"Privacy budget exceeded. Remaining: {self.session_limit - self.privacy_budget_consumed}")
        
        # Convert data to tensor if needed
        if isinstance(data, (list, np.ndarray)):
            tensor_data = torch.tensor(data, dtype=torch.float32, device=self.device)
        elif isinstance(data, torch.Tensor):
            tensor_data = data.clone().to(self.device)
        else:
            # Handle text data by generating embeddings
            tensor_data = self._text_to_embeddings(data)
        
        # Apply L2 norm clipping
        clipped_data = self._clip_l2_norm(tensor_data, clipping_norm)
        
        # Calculate noise scale
        noise_scale = self._calculate_noise_scale(epsilon, delta, clipping_norm, noise_multiplier)
        
        # Add Gaussian noise
        noise = torch.normal(mean=0, std=noise_scale, size=clipped_data.shape, device=self.device)
        noisy_data = clipped_data + noise
        
        # Update privacy budget
        self._update_privacy_budget(epsilon, delta)
        
        # Record query
        self._record_query(epsilon, delta, noise_scale, clipping_norm)
        
        return {
            'noisy_data': noisy_data.cpu().numpy(),
            'original_shape': tensor_data.shape,
            'noise_scale': noise_scale,
            'epsilon_used': epsilon,
            'delta_used': delta,
            'clipping_norm': clipping_norm,
            'privacy_budget_remaining': self.session_limit - self.privacy_budget_consumed
        }
    
    def _clip_l2_norm(self, data: torch.Tensor, clipping_norm: float) -> torch.Tensor:
        """
        Apply L2 norm clipping to bound sensitivity.
        
        Args:
            data: Input tensor
            clipping_norm: L2 norm bound
            
        Returns:
            torch.Tensor: Clipped tensor
        """
        if data.dim() == 1:
            # Single vector
            norm = torch.norm(data, p=2)
            if norm > clipping_norm:
                data = data * (clipping_norm / norm)
        else:
            # Batch of vectors
            norms = torch.norm(data, p=2, dim=-1, keepdim=True)
            clip_coef = torch.minimum(torch.ones_like(norms), clipping_norm / norms)
            data = data * clip_coef
        
        return data
    
    def _calculate_noise_scale(self, epsilon: float, delta: float, 
                              sensitivity: float, noise_multiplier: float) -> float:
        """
        Calculate the standard deviation for Gaussian noise.
        
        Args:
            epsilon: Privacy parameter
            delta: Privacy parameter  
            sensitivity: L2 sensitivity (clipping norm)
            noise_multiplier: Additional noise multiplier
            
        Returns:
            float: Noise standard deviation
        """
        # For Gaussian mechanism: σ ≥ sensitivity * √(2 * ln(1.25/δ)) / ε
        if delta <= 0 or delta >= 1:
            raise ValueError(f"Delta must be in (0, 1), got {delta}")
        
        if epsilon <= 0:
            raise ValueError(f"Epsilon must be positive, got {epsilon}")
        
        # Calculate minimum noise scale for (ε, δ)-DP
        min_noise_scale = sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon
        
        # Apply additional noise multiplier for extra safety
        actual_noise_scale = min_noise_scale * noise_multiplier
        
        return actual_noise_scale
    
    def _text_to_embeddings(self, text: str) -> torch.Tensor:
        """
        Convert text to embeddings for noise addition.
        
        Args:
            text: Input text
            
        Returns:
            torch.Tensor: Text embeddings
        """
        # This is a simplified embedding generation
        # In practice, this would use the edge model's embeddings
        
        # Create a simple hash-based embedding for demonstration
        hash_val = hash(text)
        embedding_dim = 768  # Standard BERT dimension
        
        # Generate deterministic but random-like embeddings based on text hash
        np.random.seed(abs(hash_val) % (2**32))
        embeddings = np.random.normal(0, 1, embedding_dim)
        
        return torch.tensor(embeddings, dtype=torch.float32, device=self.device)
    
    def _check_privacy_budget(self, epsilon: float) -> bool:
        """
        Check if adding epsilon would exceed privacy budget.
        
        Args:
            epsilon: Requested privacy parameter
            
        Returns:
            bool: True if budget allows, False otherwise
        """
        return self.privacy_budget_consumed + epsilon <= self.session_limit
    
    def _update_privacy_budget(self, epsilon: float, delta: float):
        """
        Update privacy budget after query.
        
        Args:
            epsilon: Used privacy parameter
            delta: Used privacy parameter
        """
        self.privacy_budget_consumed += epsilon
        self.query_count += 1
        
        logger.debug(f"Privacy budget updated: {self.privacy_budget_consumed:.3f}/{self.session_limit}")
    
    def _record_query(self, epsilon: float, delta: float, noise_scale: float, clipping_norm: float):
        """
        Record query details for audit purposes.
        
        Args:
            epsilon: Privacy parameter used
            delta: Privacy parameter used
            noise_scale: Noise scale applied
            clipping_norm: Clipping norm used
        """
        query_record = {
            'timestamp': datetime.now().isoformat(),
            'epsilon': epsilon,
            'delta': delta,
            'noise_scale': noise_scale,
            'clipping_norm': clipping_norm,
            'cumulative_epsilon': self.privacy_budget_consumed,
            'query_number': self.query_count
        }
        
        self.query_history.append(query_record)
        
        # Keep only recent queries (last 1000)
        if len(self.query_history) > 1000:
            self.query_history = self.query_history[-1000:]
    
    def get_privacy_parameters(self, sensitivity_level: str) -> Dict[str, float]:
        """
        Get appropriate privacy parameters based on sensitivity level.
        
        Args:
            sensitivity_level: 'general' or 'high_sensitivity'
            
        Returns:
            dict: Privacy parameters
        """
        if sensitivity_level == 'high_sensitivity':
            return self.high_sensitivity_params.copy()
        else:
            return self.general_params.copy()
    
    def estimate_privacy_cost(self, num_queries: int, sensitivity_level: str = 'general') -> Dict[str, float]:
        """
        Estimate privacy cost for multiple queries.
        
        Args:
            num_queries: Number of planned queries
            sensitivity_level: Sensitivity level for parameter selection
            
        Returns:
            dict: Privacy cost estimation
        """
        params = self.get_privacy_parameters(sensitivity_level)
        epsilon_per_query = params.get('epsilon', 1.0)
        
        total_epsilon = num_queries * epsilon_per_query
        remaining_budget = self.session_limit - self.privacy_budget_consumed
        
        return {
            'total_epsilon_needed': total_epsilon,
            'epsilon_per_query': epsilon_per_query,
            'current_budget_used': self.privacy_budget_consumed,
            'remaining_budget': remaining_budget,
            'queries_possible': int(remaining_budget / epsilon_per_query) if epsilon_per_query > 0 else 0,
            'budget_sufficient': total_epsilon <= remaining_budget
        }
    
    def get_privacy_accountant_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive privacy accountant report.
        
        Returns:
            dict: Privacy accounting details
        """
        session_duration = datetime.now() - self.session_start
        
        # Calculate query rate
        queries_per_minute = self.query_count / (session_duration.total_seconds() / 60) if session_duration.total_seconds() > 0 else 0
        
        # Analyze query distribution
        epsilon_values = [q['epsilon'] for q in self.query_history]
        
        report = {
            'session_info': {
                'session_start': self.session_start.isoformat(),
                'session_duration_minutes': session_duration.total_seconds() / 60,
                'total_queries': self.query_count,
                'queries_per_minute': queries_per_minute
            },
            'privacy_budget': {
                'total_budget': self.session_limit,
                'consumed_budget': self.privacy_budget_consumed,
                'remaining_budget': self.session_limit - self.privacy_budget_consumed,
                'budget_utilization': self.privacy_budget_consumed / self.session_limit
            },
            'query_statistics': {
                'min_epsilon': min(epsilon_values) if epsilon_values else 0,
                'max_epsilon': max(epsilon_values) if epsilon_values else 0,
                'avg_epsilon': sum(epsilon_values) / len(epsilon_values) if epsilon_values else 0,
                'total_epsilon': sum(epsilon_values)
            },
            'rate_limiting': {
                'rate_limit': self.rate_limit,
                'current_rate': queries_per_minute,
                'rate_compliant': queries_per_minute <= self.rate_limit
            },
            'recent_queries': self.query_history[-10:] if self.query_history else []
        }
        
        return report
    
    def reset_privacy_budget(self):
        """Reset privacy budget for new session."""
        self.privacy_budget_consumed = 0.0
        self.query_count = 0
        self.session_start = datetime.now()
        self.query_history = []
        
        logger.info("Privacy budget reset for new session")
    
    def compose_privacy_parameters(self, epsilons: List[float], deltas: List[float]) -> Tuple[float, float]:
        """
        Compose privacy parameters using composition theorems.
        
        Args:
            epsilons: List of epsilon values
            deltas: List of delta values
            
        Returns:
            tuple: Composed (epsilon, delta)
        """
        # Simple composition (conservative)
        composed_epsilon = sum(epsilons)
        composed_delta = sum(deltas)
        
        # Apply advanced composition if available
        if len(epsilons) > 1:
            # Advanced composition theorem (simplified)
            k = len(epsilons)
            max_epsilon = max(epsilons)
            
            if all(eps == max_epsilon for eps in epsilons):
                # Homogeneous case - can use advanced composition
                composed_epsilon = max_epsilon * math.sqrt(2 * k * math.log(1 / min(deltas)))
                composed_delta = k * max(deltas)
        
        return composed_epsilon, composed_delta
    
    def validate_dp_parameters(self, epsilon: float, delta: float) -> bool:
        """
        Validate differential privacy parameters.
        
        Args:
            epsilon: Privacy parameter
            delta: Privacy parameter
            
        Returns:
            bool: True if parameters are valid
        """
        if epsilon <= 0:
            logger.error(f"Epsilon must be positive, got {epsilon}")
            return False
        
        if delta <= 0 or delta >= 1:
            logger.error(f"Delta must be in (0, 1), got {delta}")
            return False
        
        if epsilon > 10:
            logger.warning(f"Epsilon {epsilon} is quite large, privacy protection may be weak")
        
        if delta > 1e-3:
            logger.warning(f"Delta {delta} is quite large, privacy protection may be weak")
        
        return True
    
    def cleanup(self):
        """Cleanup resources and save privacy accounting logs."""
        try:
            # Save privacy accounting report
            final_report = self.get_privacy_accountant_report()
            
            # Log final privacy consumption
            logger.info(f"Session completed: {self.privacy_budget_consumed:.3f}/{self.session_limit} privacy budget used")
            logger.info(f"Total queries: {self.query_count}")
            
            # Clear sensitive data
            self.query_history.clear()
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

=======
version https://git-lfs.github.com/spec/v1
oid sha256:df0574840a6bbf063d8e8aa8831ac37cc0e6617bc7c02280e55fcd786848f070
size 15132
>>>>>>> 9676c3e (ya toh aar ya toh par)
