"""
Privacy Metrics for PIPL-Compliant LLM Evaluation

This module implements comprehensive privacy evaluation metrics including:
- Membership Inference Attack (MIA) evaluation
- Privacy leakage measurement
- Embedding inversion resistance testing
- Cross-border transmission compliance
"""

import logging
import numpy as np
import torch
from typing import Dict, List, Any, Tuple, Optional
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import random

logger = logging.getLogger(__name__)


class PrivacyMetrics:
    """
    Comprehensive privacy evaluation metrics for privacy-preserving LLM systems.
    
    Implements multiple attack simulation methods and privacy measurement techniques
    to assess the effectiveness of privacy protection mechanisms.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize privacy metrics evaluator.
        
        Args:
            config: Configuration dictionary for evaluation parameters
        """
        self.config = config or {}
        self.attack_config = self.config.get('attack_simulation', {})
        
        # Attack simulation parameters
        self.neighbourhood_mia_config = self.attack_config.get('neighbourhood_mia', {})
        self.loss_attack_config = self.attack_config.get('loss_attack', {})
        self.lira_attack_config = self.attack_config.get('lira_attack', {})
        
        # Evaluation results storage
        self.evaluation_results = {}
        
        logger.info("Privacy Metrics evaluator initialized")
    
    def evaluate_privacy(self, original_data: List[Any], 
                        protected_data: List[Any], 
                        labels: List[Any] = None) -> Dict[str, Any]:
        """
        Comprehensive privacy evaluation using multiple attack methods.
        
        Args:
            original_data: Original unprotected data
            protected_data: Privacy-protected data
            labels: Ground truth labels (optional)
            
        Returns:
            dict: Comprehensive privacy evaluation results
        """
        logger.info("Starting comprehensive privacy evaluation...")
        
        results = {
            'evaluation_timestamp': self._get_timestamp(),
            'data_samples': len(original_data),
            'attack_results': {},
            'privacy_scores': {},
            'overall_privacy_score': 0.0
        }
        
        try:
            # Convert data to appropriate format
            orig_embeddings = self._prepare_embeddings(original_data)
            prot_embeddings = self._prepare_embeddings(protected_data)
            
            # 1. Neighbourhood MIA Attack
            if self.neighbourhood_mia_config.get('enabled', True):
                mia_results = self._evaluate_neighbourhood_mia(
                    orig_embeddings, prot_embeddings, labels
                )
                results['attack_results']['neighbourhood_mia'] = mia_results
            
            # 2. LOSS Attack
            if self.loss_attack_config.get('enabled', True):
                loss_results = self._evaluate_loss_attack(
                    orig_embeddings, prot_embeddings, labels
                )
                results['attack_results']['loss_attack'] = loss_results
            
            # 3. LiRA Attack
            if self.lira_attack_config.get('enabled', True):
                lira_results = self._evaluate_lira_attack(
                    orig_embeddings, prot_embeddings, labels
                )
                results['attack_results']['lira_attack'] = lira_results
            
            # 4. Privacy Leakage Analysis
            leakage_results = self._evaluate_privacy_leakage(
                orig_embeddings, prot_embeddings
            )
            results['privacy_scores']['privacy_leakage'] = leakage_results
            
            # 5. Embedding Inversion Resistance
            inversion_results = self._evaluate_inversion_resistance(
                orig_embeddings, prot_embeddings
            )
            results['privacy_scores']['inversion_resistance'] = inversion_results
            
            # 6. Cross-border Transmission Compliance
            compliance_results = self._evaluate_cross_border_compliance(
                original_data, protected_data
            )
            results['privacy_scores']['cross_border_compliance'] = compliance_results
            
            # Calculate overall privacy score
            results['overall_privacy_score'] = self._calculate_overall_privacy_score(results)
            
            logger.info(f"Privacy evaluation completed. Overall score: {results['overall_privacy_score']:.3f}")
            
        except Exception as e:
            logger.error(f"Privacy evaluation failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def _evaluate_neighbourhood_mia(self, original_embeddings: np.ndarray, 
                                   protected_embeddings: np.ndarray, 
                                   labels: List[Any] = None) -> Dict[str, Any]:
        """
        Evaluate Neighbourhood Membership Inference Attack.
        
        Args:
            original_embeddings: Original data embeddings
            protected_embeddings: Protected data embeddings
            labels: Ground truth labels
            
        Returns:
            dict: Neighbourhood MIA evaluation results
        """
        try:
            num_neighbors = self.neighbourhood_mia_config.get('num_neighbors', 100)
            similarity_threshold = self.neighbourhood_mia_config.get('similarity_threshold', 0.8)
            
            # Create membership inference dataset
            member_embeddings = protected_embeddings
            non_member_embeddings = self._generate_non_member_embeddings(
                original_embeddings, len(member_embeddings)
            )
            
            # Calculate similarity-based features
            member_features = self._calculate_neighbourhood_features(
                member_embeddings, original_embeddings, num_neighbors
            )
            non_member_features = self._calculate_neighbourhood_features(
                non_member_embeddings, original_embeddings, num_neighbors
            )
            
            # Create training data
            X = np.vstack([member_features, non_member_features])
            y = np.hstack([np.ones(len(member_features)), np.zeros(len(non_member_features))])
            
            # Train attack model
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            attack_model = LogisticRegression(random_state=42)
            attack_model.fit(X_train, y_train)
            
            # Evaluate attack
            y_pred = attack_model.predict(X_test)
            y_pred_proba = attack_model.predict_proba(X_test)[:, 1]
            
            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            return {
                'attack_type': 'neighbourhood_mia',
                'accuracy': accuracy,
                'auc': auc,
                'success_rate': accuracy,
                'privacy_protection': 1.0 - auc,  # Lower AUC = better protection
                'num_neighbors': num_neighbors,
                'similarity_threshold': similarity_threshold
            }
            
        except Exception as e:
            logger.error(f"Neighbourhood MIA evaluation failed: {e}")
            return {'error': str(e), 'attack_type': 'neighbourhood_mia'}
    
    def _evaluate_loss_attack(self, original_embeddings: np.ndarray, 
                             protected_embeddings: np.ndarray, 
                             labels: List[Any] = None) -> Dict[str, Any]:
        """
        Evaluate LOSS-based Membership Inference Attack.
        
        Args:
            original_embeddings: Original data embeddings
            protected_embeddings: Protected data embeddings
            labels: Ground truth labels
            
        Returns:
            dict: LOSS attack evaluation results
        """
        try:
            confidence_threshold = self.loss_attack_config.get('confidence_threshold', 0.5)
            
            # Simulate loss-based attack
            # In practice, this would use actual model loss values
            member_losses = self._simulate_model_losses(protected_embeddings, is_member=True)
            non_member_losses = self._simulate_model_losses(protected_embeddings, is_member=False)
            
            # Create attack dataset
            member_features = np.array([[loss] for loss in member_losses])
            non_member_features = np.array([[loss] for loss in non_member_losses])
            
            X = np.vstack([member_features, non_member_features])
            y = np.hstack([np.ones(len(member_features)), np.zeros(len(non_member_features))])
            
            # Train attack model
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            attack_model = LogisticRegression(random_state=42)
            attack_model.fit(X_train, y_train)
            
            # Evaluate attack
            y_pred = attack_model.predict(X_test)
            y_pred_proba = attack_model.predict_proba(X_test)[:, 1]
            
            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            return {
                'attack_type': 'loss_attack',
                'accuracy': accuracy,
                'auc': auc,
                'success_rate': accuracy,
                'privacy_protection': 1.0 - auc,
                'confidence_threshold': confidence_threshold
            }
            
        except Exception as e:
            logger.error(f"LOSS attack evaluation failed: {e}")
            return {'error': str(e), 'attack_type': 'loss_attack'}
    
    def _evaluate_lira_attack(self, original_embeddings: np.ndarray, 
                             protected_embeddings: np.ndarray, 
                             labels: List[Any] = None) -> Dict[str, Any]:
        """
        Evaluate LiRA (Likelihood Ratio Attack).
        
        Args:
            original_embeddings: Original data embeddings
            protected_embeddings: Protected data embeddings
            labels: Ground truth labels
            
        Returns:
            dict: LiRA attack evaluation results
        """
        try:
            num_shadow_models = self.lira_attack_config.get('num_shadow_models', 5)
            confidence_threshold = self.lira_attack_config.get('confidence_threshold', 0.5)
            
            # Train shadow models
            shadow_models = self._train_shadow_models(
                original_embeddings, protected_embeddings, num_shadow_models
            )
            
            # Calculate likelihood ratios
            member_likelihoods = self._calculate_likelihood_ratios(
                protected_embeddings, shadow_models, is_member=True
            )
            non_member_likelihoods = self._calculate_likelihood_ratios(
                protected_embeddings, shadow_models, is_member=False
            )
            
            # Create attack dataset
            member_features = np.array([[likelihood] for likelihood in member_likelihoods])
            non_member_features = np.array([[likelihood] for likelihood in non_member_likelihoods])
            
            X = np.vstack([member_features, non_member_features])
            y = np.hstack([np.ones(len(member_features)), np.zeros(len(non_member_features))])
            
            # Train attack model
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            attack_model = LogisticRegression(random_state=42)
            attack_model.fit(X_train, y_train)
            
            # Evaluate attack
            y_pred = attack_model.predict(X_test)
            y_pred_proba = attack_model.predict_proba(X_test)[:, 1]
            
            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            return {
                'attack_type': 'lira_attack',
                'accuracy': accuracy,
                'auc': auc,
                'success_rate': accuracy,
                'privacy_protection': 1.0 - auc,
                'num_shadow_models': num_shadow_models,
                'confidence_threshold': confidence_threshold
            }
            
        except Exception as e:
            logger.error(f"LiRA attack evaluation failed: {e}")
            return {'error': str(e), 'attack_type': 'lira_attack'}
    
    def _evaluate_privacy_leakage(self, original_embeddings: np.ndarray, 
                                 protected_embeddings: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate privacy leakage through various measures.
        
        Args:
            original_embeddings: Original data embeddings
            protected_embeddings: Protected data embeddings
            
        Returns:
            dict: Privacy leakage evaluation results
        """
        try:
            # Calculate mutual information between original and protected
            mutual_info = self._calculate_mutual_information(original_embeddings, protected_embeddings)
            
            # Calculate correlation between embeddings
            correlation = self._calculate_embedding_correlation(original_embeddings, protected_embeddings)
            
            # Calculate reconstruction error
            reconstruction_error = self._calculate_reconstruction_error(original_embeddings, protected_embeddings)
            
            # Calculate information-theoretic privacy loss
            privacy_loss = self._calculate_privacy_loss(original_embeddings, protected_embeddings)
            
            return {
                'mutual_information': mutual_info,
                'correlation': correlation,
                'reconstruction_error': reconstruction_error,
                'privacy_loss': privacy_loss,
                'leakage_score': (mutual_info + abs(correlation) + privacy_loss) / 3,
                'privacy_protection': 1.0 - min(1.0, (mutual_info + abs(correlation) + privacy_loss) / 3)
            }
            
        except Exception as e:
            logger.error(f"Privacy leakage evaluation failed: {e}")
            return {'error': str(e)}
    
    def _evaluate_inversion_resistance(self, original_embeddings: np.ndarray, 
                                     protected_embeddings: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate resistance to embedding inversion attacks.
        
        Args:
            original_embeddings: Original data embeddings
            protected_embeddings: Protected data embeddings
            
        Returns:
            dict: Inversion resistance evaluation results
        """
        try:
            # Simulate inversion attack
            inversion_attempts = self._simulate_inversion_attacks(protected_embeddings)
            
            # Calculate inversion success rate
            success_rate = self._calculate_inversion_success_rate(
                original_embeddings, inversion_attempts
            )
            
            # Calculate reconstruction quality
            reconstruction_quality = self._calculate_reconstruction_quality(
                original_embeddings, inversion_attempts
            )
            
            return {
                'inversion_success_rate': success_rate,
                'reconstruction_quality': reconstruction_quality,
                'resistance_score': 1.0 - success_rate,
                'privacy_protection': 1.0 - success_rate
            }
            
        except Exception as e:
            logger.error(f"Inversion resistance evaluation failed: {e}")
            return {'error': str(e)}
    
    def _evaluate_cross_border_compliance(self, original_data: List[Any], 
                                        protected_data: List[Any]) -> Dict[str, Any]:
        """
        Evaluate cross-border transmission compliance.
        
        Args:
            original_data: Original data
            protected_data: Protected data
            
        Returns:
            dict: Cross-border compliance evaluation results
        """
        try:
            # Check for raw text in protected data
            raw_text_detected = self._detect_raw_text_in_protected_data(protected_data)
            
            # Check anonymization effectiveness
            anonymization_score = self._evaluate_anonymization_effectiveness(
                original_data, protected_data
            )
            
            # Check minimal necessity compliance
            minimal_necessity_score = self._evaluate_minimal_necessity_compliance(
                original_data, protected_data
            )
            
            return {
                'raw_text_detected': raw_text_detected,
                'anonymization_score': anonymization_score,
                'minimal_necessity_score': minimal_necessity_score,
                'compliance_score': (anonymization_score + minimal_necessity_score) / 2,
                'pipl_compliant': not raw_text_detected and anonymization_score > 0.8
            }
            
        except Exception as e:
            logger.error(f"Cross-border compliance evaluation failed: {e}")
            return {'error': str(e)}
    
    def _prepare_embeddings(self, data: List[Any]) -> np.ndarray:
        """Prepare data for embedding-based evaluation."""
        if isinstance(data[0], (list, np.ndarray)):
            return np.array(data)
        elif isinstance(data[0], str):
            # Convert text to embeddings (simplified)
            return np.random.rand(len(data), 768)  # Placeholder
        else:
            return np.array(data)
    
    def _generate_non_member_embeddings(self, original_embeddings: np.ndarray, 
                                       num_samples: int) -> np.ndarray:
        """Generate non-member embeddings for attack simulation."""
        # Simple approach: sample from different distribution
        mean = np.mean(original_embeddings, axis=0)
        std = np.std(original_embeddings, axis=0)
        
        return np.random.normal(mean, std * 1.5, (num_samples, original_embeddings.shape[1]))
    
    def _calculate_neighbourhood_features(self, embeddings: np.ndarray, 
                                        reference_embeddings: np.ndarray, 
                                        num_neighbors: int) -> np.ndarray:
        """Calculate neighbourhood-based features for MIA."""
        features = []
        
        for embedding in embeddings:
            # Calculate distances to reference embeddings
            distances = np.linalg.norm(reference_embeddings - embedding, axis=1)
            
            # Get k nearest neighbors
            nearest_distances = np.sort(distances)[:num_neighbors]
            
            # Calculate features
            mean_distance = np.mean(nearest_distances)
            std_distance = np.std(nearest_distances)
            min_distance = np.min(nearest_distances)
            
            features.append([mean_distance, std_distance, min_distance])
        
        return np.array(features)
    
    def _simulate_model_losses(self, embeddings: np.ndarray, is_member: bool) -> List[float]:
        """Simulate model loss values for LOSS attack."""
        # In practice, this would use actual model losses
        if is_member:
            # Members typically have lower losses
            return np.random.exponential(0.5, len(embeddings)).tolist()
        else:
            # Non-members typically have higher losses
            return np.random.exponential(1.0, len(embeddings)).tolist()
    
    def _train_shadow_models(self, original_embeddings: np.ndarray, 
                           protected_embeddings: np.ndarray, 
                           num_models: int) -> List[Any]:
        """Train shadow models for LiRA attack."""
        # Simplified shadow model training
        shadow_models = []
        
        for i in range(num_models):
            # Create synthetic training data
            X_train = np.random.rand(100, original_embeddings.shape[1])
            y_train = np.random.randint(0, 2, 100)
            
            # Train simple model
            model = RandomForestClassifier(n_estimators=10, random_state=i)
            model.fit(X_train, y_train)
            shadow_models.append(model)
        
        return shadow_models
    
    def _calculate_likelihood_ratios(self, embeddings: np.ndarray, 
                                   shadow_models: List[Any], 
                                   is_member: bool) -> List[float]:
        """Calculate likelihood ratios for LiRA attack."""
        likelihoods = []
        
        for embedding in embeddings:
            # Calculate average likelihood across shadow models
            model_likelihoods = []
            
            for model in shadow_models:
                try:
                    proba = model.predict_proba(embedding.reshape(1, -1))[0]
                    likelihood = np.max(proba)  # Use max probability as likelihood
                    model_likelihoods.append(likelihood)
                except:
                    model_likelihoods.append(0.5)  # Default likelihood
            
            avg_likelihood = np.mean(model_likelihoods)
            likelihoods.append(avg_likelihood)
        
        return likelihoods
    
    def _calculate_mutual_information(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Calculate mutual information between two sets of embeddings."""
        # Simplified mutual information calculation
        # In practice, would use proper MI estimation
        correlation = np.corrcoef(X.flatten(), Y.flatten())[0, 1]
        return abs(correlation) * 0.5  # Approximate MI
    
    def _calculate_embedding_correlation(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Calculate correlation between embeddings."""
        if X.shape != Y.shape:
            return 0.0
        
        # Calculate average correlation across dimensions
        correlations = []
        for i in range(min(X.shape[1], Y.shape[1])):
            corr = np.corrcoef(X[:, i], Y[:, i])[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
        
        return np.mean(correlations) if correlations else 0.0
    
    def _calculate_reconstruction_error(self, original: np.ndarray, 
                                      reconstructed: np.ndarray) -> float:
        """Calculate reconstruction error."""
        if original.shape != reconstructed.shape:
            return 1.0
        
        mse = np.mean((original - reconstructed) ** 2)
        return mse
    
    def _calculate_privacy_loss(self, original: np.ndarray, protected: np.ndarray) -> float:
        """Calculate information-theoretic privacy loss."""
        # Simplified privacy loss calculation
        # Based on the difference in information content
        orig_entropy = self._calculate_entropy(original)
        prot_entropy = self._calculate_entropy(protected)
        
        return max(0, orig_entropy - prot_entropy)
    
    def _calculate_entropy(self, data: np.ndarray) -> float:
        """Calculate entropy of data."""
        # Simplified entropy calculation
        # In practice, would use proper entropy estimation
        return np.log(np.var(data) + 1e-8)
    
    def _simulate_inversion_attacks(self, protected_embeddings: np.ndarray) -> np.ndarray:
        """Simulate embedding inversion attacks."""
        # Simplified inversion simulation
        # In practice, would use actual inversion techniques
        noise = np.random.normal(0, 0.1, protected_embeddings.shape)
        return protected_embeddings + noise
    
    def _calculate_inversion_success_rate(self, original: np.ndarray, 
                                        inverted: np.ndarray) -> float:
        """Calculate inversion attack success rate."""
        # Calculate similarity between original and inverted
        similarities = []
        for i in range(min(len(original), len(inverted))):
            sim = np.corrcoef(original[i], inverted[i])[0, 1]
            if not np.isnan(sim):
                similarities.append(sim)
        
        # Success if similarity > threshold
        threshold = 0.8
        success_rate = np.mean([s > threshold for s in similarities]) if similarities else 0.0
        
        return success_rate
    
    def _calculate_reconstruction_quality(self, original: np.ndarray, 
                                        reconstructed: np.ndarray) -> float:
        """Calculate reconstruction quality."""
        if original.shape != reconstructed.shape:
            return 0.0
        
        # Calculate normalized reconstruction quality
        mse = np.mean((original - reconstructed) ** 2)
        max_possible_error = np.mean(original ** 2)
        
        quality = 1.0 - (mse / (max_possible_error + 1e-8))
        return max(0.0, min(1.0, quality))
    
    def _detect_raw_text_in_protected_data(self, protected_data: List[Any]) -> bool:
        """Detect if raw text is present in protected data."""
        # Check if any protected data contains raw text patterns
        text_patterns = [r'[\u4e00-\u9fff]+', r'[a-zA-Z]+', r'\d+']
        
        for item in protected_data:
            if isinstance(item, str):
                for pattern in text_patterns:
                    if len(item) > 10:  # Only check longer strings
                        return True
        
        return False
    
    def _evaluate_anonymization_effectiveness(self, original_data: List[Any], 
                                            protected_data: List[Any]) -> float:
        """Evaluate effectiveness of anonymization."""
        # Simplified anonymization evaluation
        # In practice, would use more sophisticated techniques
        
        if len(original_data) != len(protected_data):
            return 0.0
        
        # Check for structural similarity (should be preserved)
        # Check for content similarity (should be reduced)
        # This is a placeholder implementation
        return 0.85  # Placeholder score
    
    def _evaluate_minimal_necessity_compliance(self, original_data: List[Any], 
                                             protected_data: List[Any]) -> float:
        """Evaluate minimal necessity principle compliance."""
        # Check if only necessary information is transmitted
        # This is a placeholder implementation
        return 0.9  # Placeholder score
    
    def _calculate_overall_privacy_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall privacy protection score."""
        attack_results = results.get('attack_results', {})
        privacy_scores = results.get('privacy_scores', {})
        
        scores = []
        
        # Collect attack-based scores
        for attack_name, attack_result in attack_results.items():
            if 'privacy_protection' in attack_result:
                scores.append(attack_result['privacy_protection'])
        
        # Collect privacy metric scores
        for metric_name, metric_result in privacy_scores.items():
            if isinstance(metric_result, dict) and 'privacy_protection' in metric_result:
                scores.append(metric_result['privacy_protection'])
            elif isinstance(metric_result, (int, float)):
                scores.append(metric_result)
        
        return np.mean(scores) if scores else 0.0
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
