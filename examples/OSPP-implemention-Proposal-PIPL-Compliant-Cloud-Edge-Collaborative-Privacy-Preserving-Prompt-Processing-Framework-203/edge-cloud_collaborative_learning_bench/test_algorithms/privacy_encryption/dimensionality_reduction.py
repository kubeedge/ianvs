"""
Dimensionality Reduction for Privacy Protection

This module implements Johnson-Lindenstrauss projection and other dimensionality
reduction techniques to compress embeddings while preserving semantic information
for privacy-preserving LLM inference.
"""

import logging
import numpy as np
import torch
from typing import Dict, List, Any, Tuple, Optional
from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import PCA
import math

logger = logging.getLogger(__name__)


class DimensionalityReduction:
    """
    Implements dimensionality reduction techniques for privacy-preserving embeddings.
    
    Uses Johnson-Lindenstrauss projection and other methods to compress high-dimensional
    embeddings while preserving semantic information and privacy properties.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize dimensionality reduction module.
        
        Args:
            config: Configuration dictionary containing reduction parameters
        """
        self.config = config
        self.anonymization_config = config.get('anonymization', {})
        
        # Reduction parameters
        self.projection_method = self.anonymization_config.get('projection_method', 'johnson_lindenstrauss')
        self.target_dims = self.anonymization_config.get('target_dims', 64)
        self.preserve_ratio = self.anonymization_config.get('preserve_ratio', 0.8)
        
        # Initialize projection matrices
        self.projection_matrix = None
        self.pca_model = None
        self.is_fitted = False
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info("Dimensionality Reduction module initialized")
    
    def reduce_dimensions(self, data: Any, method: Optional[str] = None) -> Dict[str, Any]:
        """
        Reduce dimensions of input data.
        
        Args:
            data: Input data (embeddings, vectors, or text)
            method: Reduction method to use (overrides config)
            
        Returns:
            dict: Reduced data with metadata
        """
        if data is None:
            return {
                'reduced_data': None,
                'original_dims': 0,
                'reduced_dims': 0,
                'compression_ratio': 0.0,
                'method_used': 'none'
            }
        
        # Convert data to numpy array
        if isinstance(data, torch.Tensor):
            data_array = data.cpu().numpy()
        elif isinstance(data, list):
            data_array = np.array(data)
        elif isinstance(data, str):
            # Convert text to embeddings first
            data_array = self._text_to_embeddings(data)
        else:
            data_array = np.array(data)
        
        # Ensure 2D array
        if data_array.ndim == 1:
            data_array = data_array.reshape(1, -1)
        
        original_dims = data_array.shape[-1]
        
        # Select reduction method
        reduction_method = method or self.projection_method
        
        # Apply dimensionality reduction
        if reduction_method == 'johnson_lindenstrauss':
            reduced_data, projection_info = self._johnson_lindenstrauss_projection(data_array)
        elif reduction_method == 'pca':
            reduced_data, projection_info = self._pca_projection(data_array)
        elif reduction_method == 'random_projection':
            reduced_data, projection_info = self._random_projection(data_array)
        else:
            raise ValueError(f"Unknown reduction method: {reduction_method}")
        
        reduced_dims = reduced_data.shape[-1]
        compression_ratio = reduced_dims / original_dims if original_dims > 0 else 0.0
        
        return {
            'reduced_data': reduced_data,
            'original_dims': original_dims,
            'reduced_dims': reduced_dims,
            'compression_ratio': compression_ratio,
            'method_used': reduction_method,
            'projection_info': projection_info,
            'preservation_quality': self._estimate_preservation_quality(data_array, reduced_data)
        }
    
    def _johnson_lindenstrauss_projection(self, data: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply Johnson-Lindenstrauss projection.
        
        Args:
            data: Input data array
            
        Returns:
            tuple: (reduced_data, projection_info)
        """
        n_samples, n_features = data.shape
        
        # Calculate optimal dimensions based on Johnson-Lindenstrauss lemma
        optimal_dims = self._calculate_jl_dimensions(n_samples, n_features)
        target_dims = min(self.target_dims, optimal_dims, n_features)
        
        # Initialize or reuse projection matrix
        if (self.projection_matrix is None or 
            self.projection_matrix.shape != (n_features, target_dims)):
            
            # Generate random projection matrix
            self.projection_matrix = self._generate_jl_projection_matrix(n_features, target_dims)
        
        # Apply projection
        reduced_data = data @ self.projection_matrix
        
        projection_info = {
            'method': 'johnson_lindenstrauss',
            'original_dims': n_features,
            'reduced_dims': target_dims,
            'projection_matrix_shape': self.projection_matrix.shape,
            'theoretical_bound': self._calculate_jl_bound(n_samples, target_dims)
        }
        
        return reduced_data, projection_info
    
    def _pca_projection(self, data: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply PCA projection.
        
        Args:
            data: Input data array
            
        Returns:
            tuple: (reduced_data, projection_info)
        """
        n_samples, n_features = data.shape
        target_dims = min(self.target_dims, n_features, n_samples - 1)
        
        # Initialize or fit PCA model
        if (self.pca_model is None or 
            not self.is_fitted or
            self.pca_model.n_components_ != target_dims):
            
            self.pca_model = PCA(n_components=target_dims, random_state=42)
            self.pca_model.fit(data)
            self.is_fitted = True
        
        # Apply PCA transformation
        reduced_data = self.pca_model.transform(data)
        
        # Calculate explained variance
        explained_variance_ratio = self.pca_model.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        projection_info = {
            'method': 'pca',
            'original_dims': n_features,
            'reduced_dims': target_dims,
            'explained_variance_ratio': explained_variance_ratio.tolist(),
            'cumulative_variance': cumulative_variance.tolist(),
            'total_variance_explained': cumulative_variance[-1] if len(cumulative_variance) > 0 else 0.0
        }
        
        return reduced_data, projection_info
    
    def _random_projection(self, data: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply random projection using sklearn.
        
        Args:
            data: Input data array
            
        Returns:
            tuple: (reduced_data, projection_info)
        """
        n_samples, n_features = data.shape
        target_dims = min(self.target_dims, n_features)
        
        # Use Gaussian random projection
        projector = GaussianRandomProjection(
            n_components=target_dims,
            random_state=42
        )
        
        reduced_data = projector.fit_transform(data)
        
        projection_info = {
            'method': 'random_projection',
            'original_dims': n_features,
            'reduced_dims': target_dims,
            'projection_type': 'gaussian',
            'random_state': 42
        }
        
        return reduced_data, projection_info
    
    def _calculate_jl_dimensions(self, n_samples: int, n_features: int) -> int:
        """
        Calculate optimal dimensions for Johnson-Lindenstrauss projection.
        
        Args:
            n_samples: Number of samples
            n_features: Number of features
            
        Returns:
            int: Optimal number of dimensions
        """
        # Johnson-Lindenstrauss bound: k >= 4 * ln(n) / (ε²/2 - ε³/3)
        # For ε = 0.1 (10% distortion), k >= 4 * ln(n) / (0.005 - 0.00033)
        epsilon = 0.1  # 10% distortion tolerance
        k_bound = 4 * math.log(n_samples) / (epsilon**2/2 - epsilon**3/3)
        
        # Ensure k is reasonable
        k_optimal = max(1, min(int(k_bound), n_features, self.target_dims))
        
        return k_optimal
    
    def _generate_jl_projection_matrix(self, n_features: int, n_components: int) -> np.ndarray:
        """
        Generate Johnson-Lindenstrauss projection matrix.
        
        Args:
            n_features: Number of input features
            n_components: Number of output components
            
        Returns:
            np.ndarray: Projection matrix
        """
        # Generate random matrix with Gaussian distribution
        # Scale by 1/sqrt(n_components) to maintain unit variance
        projection_matrix = np.random.normal(
            0, 1/np.sqrt(n_components), 
            (n_features, n_components)
        )
        
        return projection_matrix
    
    def _calculate_jl_bound(self, n_samples: int, n_components: int) -> float:
        """
        Calculate Johnson-Lindenstrauss theoretical bound.
        
        Args:
            n_samples: Number of samples
            n_components: Number of components
            
        Returns:
            float: Theoretical distortion bound
        """
        # JL bound: with probability 1-δ, distortion ≤ ε
        # where ε = sqrt(4 * ln(n) / k)
        delta = 0.1  # 10% failure probability
        epsilon = math.sqrt(4 * math.log(n_samples) / n_components)
        
        return epsilon
    
    def _text_to_embeddings(self, text: str) -> np.ndarray:
        """
        Convert text to embeddings for dimensionality reduction.
        
        Args:
            text: Input text
            
        Returns:
            np.ndarray: Text embeddings
        """
        # This is a simplified embedding generation
        # In practice, this would use the actual edge model's embeddings
        
        # Create a simple hash-based embedding for demonstration
        hash_val = hash(text)
        embedding_dim = 768  # Standard BERT dimension
        
        # Generate deterministic but random-like embeddings
        np.random.seed(abs(hash_val) % (2**32))
        embeddings = np.random.normal(0, 1, embedding_dim)
        
        return embeddings
    
    def _estimate_preservation_quality(self, original: np.ndarray, reduced: np.ndarray) -> Dict[str, float]:
        """
        Estimate how well the reduction preserves information.
        
        Args:
            original: Original data
            reduced: Reduced data
            
        Returns:
            dict: Quality metrics
        """
        if original.shape[0] != reduced.shape[0]:
            return {'error': 'Dimension mismatch'}
        
        # Calculate pairwise distances in original space
        from sklearn.metrics.pairwise import euclidean_distances
        
        if original.shape[0] > 1:
            orig_distances = euclidean_distances(original)
            reduced_distances = euclidean_distances(reduced)
            
            # Calculate distance preservation ratio
            non_zero_mask = orig_distances > 0
            if np.any(non_zero_mask):
                distance_ratios = reduced_distances[non_zero_mask] / orig_distances[non_zero_mask]
                mean_ratio = np.mean(distance_ratios)
                std_ratio = np.std(distance_ratios)
            else:
                mean_ratio = 1.0
                std_ratio = 0.0
        else:
            # Single sample case
            orig_norm = np.linalg.norm(original)
            reduced_norm = np.linalg.norm(reduced)
            mean_ratio = reduced_norm / orig_norm if orig_norm > 0 else 1.0
            std_ratio = 0.0
        
        # Calculate compression efficiency
        compression_ratio = reduced.shape[1] / original.shape[1]
        
        # Estimate information preservation
        # Higher mean_ratio closer to 1.0 indicates better preservation
        preservation_score = 1.0 - abs(1.0 - mean_ratio)
        
        return {
            'distance_preservation_ratio': mean_ratio,
            'distance_preservation_std': std_ratio,
            'compression_ratio': compression_ratio,
            'preservation_score': max(0.0, min(1.0, preservation_score)),
            'quality_grade': self._grade_preservation_quality(preservation_score)
        }
    
    def _grade_preservation_quality(self, score: float) -> str:
        """
        Grade preservation quality based on score.
        
        Args:
            score: Preservation quality score (0-1)
            
        Returns:
            str: Quality grade
        """
        if score >= 0.9:
            return 'excellent'
        elif score >= 0.8:
            return 'good'
        elif score >= 0.7:
            return 'fair'
        elif score >= 0.6:
            return 'poor'
        else:
            return 'very_poor'
    
    def fit_projection(self, training_data: np.ndarray, method: Optional[str] = None) -> Dict[str, Any]:
        """
        Fit projection matrix on training data.
        
        Args:
            training_data: Training data for fitting
            method: Projection method to use
            
        Returns:
            dict: Fitting results
        """
        if training_data is None or len(training_data) == 0:
            return {'error': 'No training data provided'}
        
        # Convert to numpy if needed
        if isinstance(training_data, torch.Tensor):
            training_data = training_data.cpu().numpy()
        
        # Ensure 2D
        if training_data.ndim == 1:
            training_data = training_data.reshape(1, -1)
        
        reduction_method = method or self.projection_method
        
        try:
            if reduction_method == 'pca':
                # Fit PCA model
                self.pca_model = PCA(n_components=self.target_dims, random_state=42)
                self.pca_model.fit(training_data)
                self.is_fitted = True
                
                return {
                    'method': 'pca',
                    'fitted': True,
                    'explained_variance_ratio': self.pca_model.explained_variance_ratio_.tolist(),
                    'n_components': self.pca_model.n_components_
                }
            
            elif reduction_method == 'johnson_lindenstrauss':
                # Generate projection matrix
                n_features = training_data.shape[1]
                self.projection_matrix = self._generate_jl_projection_matrix(n_features, self.target_dims)
                
                return {
                    'method': 'johnson_lindenstrauss',
                    'fitted': True,
                    'projection_matrix_shape': self.projection_matrix.shape
                }
            
            else:
                return {'error': f'Unknown method: {reduction_method}'}
                
        except Exception as e:
            logger.error(f"Failed to fit projection: {e}")
            return {'error': str(e)}
    
    def inverse_transform(self, reduced_data: np.ndarray) -> np.ndarray:
        """
        Attempt to reconstruct original data from reduced data.
        
        Note: This is only approximate and may not be possible for all methods.
        
        Args:
            reduced_data: Reduced dimension data
            
        Returns:
            np.ndarray: Reconstructed data (approximate)
        """
        if self.pca_model and self.is_fitted:
            # PCA has exact inverse transformation
            return self.pca_model.inverse_transform(reduced_data)
        
        elif self.projection_matrix is not None:
            # For random projections, use pseudo-inverse
            try:
                pseudo_inverse = np.linalg.pinv(self.projection_matrix)
                return reduced_data @ pseudo_inverse.T
            except np.linalg.LinAlgError:
                logger.warning("Cannot compute pseudo-inverse for reconstruction")
                return reduced_data
        
        else:
            logger.warning("No fitted projection model available for inverse transform")
            return reduced_data
    
    def get_reduction_statistics(self, reduction_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate statistics about the dimensionality reduction.
        
        Args:
            reduction_result: Result from reduce_dimensions
            
        Returns:
            dict: Reduction statistics
        """
        stats = {
            'compression': {
                'original_dims': reduction_result.get('original_dims', 0),
                'reduced_dims': reduction_result.get('reduced_dims', 0),
                'compression_ratio': reduction_result.get('compression_ratio', 0.0),
                'space_saved': 1.0 - reduction_result.get('compression_ratio', 0.0)
            },
            'method_info': {
                'method_used': reduction_result.get('method_used', 'unknown'),
                'projection_info': reduction_result.get('projection_info', {})
            },
            'quality_metrics': reduction_result.get('preservation_quality', {})
        }
        
        # Add method-specific statistics
        projection_info = reduction_result.get('projection_info', {})
        if projection_info.get('method') == 'pca':
            stats['pca_specific'] = {
                'total_variance_explained': projection_info.get('total_variance_explained', 0.0),
                'explained_variance_ratio': projection_info.get('explained_variance_ratio', [])
            }
        elif projection_info.get('method') == 'johnson_lindenstrauss':
            stats['jl_specific'] = {
                'theoretical_bound': projection_info.get('theoretical_bound', 0.0),
                'projection_matrix_shape': projection_info.get('projection_matrix_shape', (0, 0))
            }
        
        return stats

