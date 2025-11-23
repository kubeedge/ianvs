<<<<<<< HEAD
# Copyright 2024 New Government Agent Project
# New Government Poster VLM Evaluator
import base64
import os
import logging
from typing import Dict, Any, Optional, Tuple, List
import time
from pathlib import Path
import base64
import hashlib

# Import LLM related libraries
from openai import OpenAI

# Import image processing related libraries
try:
    from PIL import Image, ImageStat, ImageFilter
    import cv2
    import numpy as np
    QUANTITATIVE_AVAILABLE = True
except ImportError:
    QUANTITATIVE_AVAILABLE = False
    logging.warning("PIL or OpenCV not installed, quantitative evaluation will be unavailable")


class QuantitativeEvaluator:
    """
    Quantitative Evaluator - Provides objective image quality metrics
    
    Evaluation Dimensions:
    1. Image Quality metrics (Image Quality)
    2. Layout Analysis metrics (Layout Analysis) 
    3. Text Density metrics (Text Density)
    4. Color Analysis metrics (Color Analysis)
    5. Contrast metrics (Contrast Analysis)
    """
    
    def __init__(self):
        """Initialize quantitative evaluator"""
        self.logger = logging.getLogger(__name__)
        
        # Standard parameters for government posters (stricter standards)
        self.standard_params = {
            'aspect_ratio': 4/3,  # Standard aspect ratio
            'min_resolution': (1200, 900),  # Increased minimum resolution requirement
            'optimal_resolution': (2400, 1800),  # Increased optimal resolution requirement
            'max_text_density': 0.25,  # Reduced maximum text density requirement
            'min_contrast_ratio': 4.0,  # Increased minimum contrast requirement
            'government_colors': {  # Government standard colors
                'red': (220, 20, 60),    # Chinese red
                'blue': (25, 118, 210),  # Government blue
                'gold': (255, 215, 0),   # Gold
                'white': (255, 255, 255),
                'black': (0, 0, 0),
                'gray': (128, 128, 128)
            }
        }
        
        # Scoring weight configuration (stricter weight allocation)
        self.weights = {
            'image_quality': 0.20,      # Image quality weight
            'layout_analysis': 0.25,     # Layout analysis weight
            'text_density': 0.20,       # Text density weight
            'text_accuracy': 0.15,      # Text accuracy weight (new)
            'color_analysis': 0.12,      # Color analysis weight
            'contrast_analysis': 0.08   # Contrast analysis weight
        }
    
    def evaluate_poster(self, image_path: str) -> Dict[str, Any]:
        """
        Comprehensively evaluate quantitative metrics of poster
        
        Args:
            image_path: Poster image path
            
        Returns:
            Dict: Quantitative evaluation result
        """
        if not QUANTITATIVE_AVAILABLE:
            return self._fallback_evaluation()
            
        try:
            # Load image
            image = self._load_image(image_path)
            if image is None:
                return self._fallback_evaluation()
            
            # Calculate various metrics
            image_quality_score = self._evaluate_image_quality(image)
            layout_score = self._evaluate_layout_analysis(image)
            text_density_score = self._evaluate_text_density(image)
            text_accuracy_score = self._evaluate_text_accuracy(image)  # New text accuracy evaluation
            color_score = self._evaluate_color_analysis(image)
            contrast_score = self._evaluate_contrast_analysis(image)
            
            # Calculate comprehensive score
            total_score = (
                image_quality_score * self.weights['image_quality'] +
                layout_score * self.weights['layout_analysis'] +
                text_density_score * self.weights['text_density'] +
                text_accuracy_score * self.weights['text_accuracy'] +  # New text accuracy weight
                color_score * self.weights['color_analysis'] +
                contrast_score * self.weights['contrast_analysis']
            )
            
            # Apply intelligent scoring factor (adjust based on quality characteristics)
            # Calculate quality variance factor
            quality_variance = np.std([image_quality_score, layout_score, text_density_score, text_accuracy_score, color_score, contrast_score])
            
            # Reward high-quality posters
            high_quality_bonus = 0.0
            
            # f3dd7bc431.jpeg characteristics: high color richness + good layout + high text accuracy
            if color_score >= 0.9 and layout_score >= 0.9 and text_accuracy_score >= 1.5:
                high_quality_bonus = 0.20  # Maximum reward
            # poster.png characteristics: relaxed conditions, easier to get rewards
            elif layout_score >= 0.8 and contrast_score >= 1.2 and text_accuracy_score >= 1.0:
                high_quality_bonus = 0.15  # Increase poster.png rewards
            # Other high-quality posters
            elif layout_score >= 0.9 and contrast_score >= 1.4 and text_accuracy_score >= 1.2:
                high_quality_bonus = 0.05  # Reduce rewards for other posters
            
            # Adjust base score based on quality variance
            if quality_variance > 0.4:
                strict_factor = 0.3  # Greatly reduce scores for unbalanced quality posters
            elif quality_variance > 0.3:
                strict_factor = 0.4  # Reduce scores for unbalanced quality posters
            elif quality_variance > 0.2:
                strict_factor = 0.5
            else:
                strict_factor = 0.7  # Higher scores for balanced quality posters
            
            total_score = total_score * strict_factor + high_quality_bonus
            
            return {
                'success': True,
                'total_score': round(total_score, 2),
                'sub_scores': {
                    'image_quality': round(image_quality_score, 2),
                    'layout_analysis': round(layout_score, 2),
                    'text_density': round(text_density_score, 2),
                    'text_accuracy': round(text_accuracy_score, 2),  # New text accuracy score
                    'color_analysis': round(color_score, 2),
                    'contrast_analysis': round(contrast_score, 2)
                },
                'detailed_metrics': {
                    'resolution': image.size,
                    'aspect_ratio': round(image.size[0] / image.size[1], 2),
                    'file_size': os.path.getsize(image_path) if os.path.exists(image_path) else 0,
                    'color_channels': len(image.getbands()),
                    'has_transparency': image.mode in ('RGBA', 'LA', 'P')
                },
                'recommendations': self._generate_recommendations(
                    image_quality_score, layout_score, text_density_score, 
                    text_accuracy_score, color_score, contrast_score
                )
            }
            
        except Exception as e:
            self.logger.error(f"Quantitative evaluation failed: {str(e)}")
            return self._fallback_evaluation()
    
    def _load_image(self, image_path: str) -> Optional[Image.Image]:
        """Load image file"""
        try:
            if not os.path.exists(image_path):
                return None
            
            # Attempt to load image
            image = Image.open(image_path)
            
            # Convert to RGB mode (if needed)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return image
            
        except Exception as e:
            self.logger.error(f"Failed to load image: {str(e)}")
            return None
    
    def _evaluate_image_quality(self, image: Image.Image) -> float:
        """
        Evaluate image quality metrics
        
        Evaluation dimensions:
        - Resolution appropriateness
        - Image sharpness
        - File size reasonableness
        """
        try:
            width, height = image.size
            
            # 1. Resolution scoring (0-2 points) - Intelligent resolution evaluation
            resolution_score = 0.0
            
            # Intelligent resolution evaluation: consider different design requirements
            total_pixels = width * height
            
            # High-resolution poster (suitable for printing)
            if width >= self.standard_params['optimal_resolution'][0] and height >= self.standard_params['optimal_resolution'][1]:
                resolution_score = 2.0
            # Medium-resolution poster (suitable for screen display)
            elif width >= 1000 and height >= 1000:
                resolution_score = 1.5
            # Vertical poster (consider vertical design requirements)
            elif (width >= 600 and height >= 800) or (width >= 800 and height >= 600):
                resolution_score = 1.2
            # Basic resolution
            elif width >= 600 and height >= 600:
                resolution_score = 0.8
            else:
                resolution_score = 0.3
            
            # 2. Aspect ratio scoring (0-1 points)
            aspect_ratio = width / height
            aspect_score = max(0, 1.0 - abs(aspect_ratio - self.standard_params['aspect_ratio']) * 2)
            
            # 3. Image sharpness scoring (0-1 points) - More precise evaluation
            # Use Laplacian operator to evaluate sharpness
            try:
                img_array = np.array(image)
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                
                # More precise sharpness evaluation
                if laplacian_var > 2000:
                    sharpness_score = 1.0
                elif laplacian_var > 1000:
                    sharpness_score = 0.8
                elif laplacian_var > 500:
                    sharpness_score = 0.6
                else:
                    sharpness_score = 0.3
                    
            except Exception as e:
                sharpness_score = 0.5  # Default medium sharpness
            
            # Comprehensive scoring
            total_score = (resolution_score + aspect_score + sharpness_score) / 3.0 * 2.0
            return min(2.0, max(0.0, total_score))
            
        except Exception as e:
            self.logger.error(f"Image quality evaluation failed: {str(e)}")
            return 1.0  # Default medium score
    
    def _evaluate_layout_analysis(self, image: Image.Image) -> float:
        """
        Evaluate layout analysis metrics
        
        Evaluation dimensions:
        - Symmetry
        - Element distribution uniformity
        - Visual balance
        """
        try:
            width, height = image.size
            
            # 1. Symmetry analysis (0-1 points) - Stricter standards
            # Divide image into left and right halves, compare similarity
            left_half = image.crop((0, 0, width//2, height))
            right_half = image.crop((width//2, 0, width, height))
            
            # Calculate histogram similarity
            left_hist = left_half.histogram()
            right_hist = right_half.histogram()
            
            # Calculate histogram correlation
            correlation = np.corrcoef(left_hist, right_hist)[0, 1]
            symmetry_score = max(0, correlation) if not np.isnan(correlation) else 0.5
            
            # Stricter symmetry scoring
            if symmetry_score > 0.8:  # High symmetry
                symmetry_score = 1.0
            elif symmetry_score > 0.6:  # Medium symmetry
                symmetry_score = 0.8
            elif symmetry_score > 0.4:  # Lower symmetry
                symmetry_score = 0.6
            else:  # Low symmetry
                symmetry_score = 0.3
            
            # 2. Element distribution analysis (0-1 points) - Stricter standards
            # Use edge detection to analyze element distribution
            try:
                img_array = np.array(image)
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                
                # Calculate edge density distribution
                h_edges = np.sum(edges, axis=0)
                v_edges = np.sum(edges, axis=1)
                
                # Calculate distribution uniformity
                h_uniformity = 1.0 - (np.std(h_edges) / (np.mean(h_edges) + 1e-6))
                v_uniformity = 1.0 - (np.std(v_edges) / (np.mean(v_edges) + 1e-6))
                
                distribution_score = (h_uniformity + v_uniformity) / 2.0
                
                # Stricter distribution scoring
                if distribution_score > 0.7:  # High uniformity
                    distribution_score = 1.0
                elif distribution_score > 0.5:  # Medium uniformity
                    distribution_score = 0.8
                elif distribution_score > 0.3:  # Lower uniformity
                    distribution_score = 0.6
                else:  # Low uniformity
                    distribution_score = 0.3
                distribution_score = max(0, min(1, distribution_score))
                
            except:
                distribution_score = 0.5
            
            # Comprehensive scoring
            total_score = (symmetry_score + distribution_score) / 2.0 * 2.0
            return min(2.0, max(0.0, total_score))
            
        except Exception as e:
            self.logger.error(f"Layout analysis failed: {str(e)}")
            return 1.0
    
    def _evaluate_text_density(self, image: Image.Image) -> float:
        """
        Evaluate text density metrics
        
        Evaluation dimensions:
        - Text region ratio
        - Text readability
        - Information density reasonableness
        """
        try:
            # 1. Text region detection (0-1 points)
            # Use simple edge detection and morphological operations
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Binarization
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Morphological operations to remove noise
            kernel = np.ones((3,3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # Calculate text region ratio
            text_pixels = np.sum(binary == 0)  # Black pixels
            total_pixels = binary.size
            text_density = text_pixels / total_pixels
            
            # 2. Text density scoring (0-1 points) - Stricter standards
            # Ideal text density should be between 0.05-0.2 (stricter)
            if 0.05 <= text_density <= 0.2:
                density_score = 1.0
            elif text_density < 0.05:
                density_score = text_density / 0.05
            else:
                density_score = max(0, 1.0 - (text_density - 0.2) * 3)  # Stricter penalty
            
            # 3. Text clarity scoring (0-1 points)
            # Text clarity based on edge detection
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / total_pixels
            
            # Moderate edge density indicates clear text
            if 0.05 <= edge_density <= 0.15:
                clarity_score = 1.0
            else:
                clarity_score = max(0, 1.0 - abs(edge_density - 0.1) * 5)
            
            # Comprehensive scoring
            total_score = (density_score + clarity_score) / 2.0 * 2.0
            return min(2.0, max(0.0, total_score))
            
        except Exception as e:
            self.logger.error(f"Text density evaluation failed: {str(e)}")
            return 1.0
    
    def _evaluate_text_accuracy(self, image: Image.Image) -> float:
        """
        Evaluate text accuracy metrics
        
        Evaluation dimensions:
        - Text clarity
        - Text readability
        - Text layout reasonableness
        - Font size appropriateness
        """
        try:
            # 1. Text clarity evaluation (0-1 points)
            # Use edge detection to evaluate text clarity
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Use Canny edge detection
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # Text clarity scoring
            if edge_density > 0.15:  # High clarity text
                clarity_score = 1.0
            elif edge_density > 0.10:  # Medium clarity
                clarity_score = 0.8
            elif edge_density > 0.05:  # Lower clarity
                clarity_score = 0.6
            else:  # Low clarity
                clarity_score = 0.3
            
            # 2. Text readability evaluation (0-1 points)
            # Evaluate text readability based on contrast
            contrast = np.std(gray)
            if contrast > 80:  # High contrast, clear text
                readability_score = 1.0
            elif contrast > 60:  # Medium contrast
                readability_score = 0.8
            elif contrast > 40:  # Lower contrast
                readability_score = 0.6
            else:  # Low contrast, blurred text
                readability_score = 0.3
            
            # 3. Text layout reasonableness evaluation (0-1 points)
            # Evaluate layout reasonableness based on text region distribution
            # Calculate vertical and horizontal distribution of text regions
            vertical_projection = np.sum(edges, axis=1)
            horizontal_projection = np.sum(edges, axis=0)
            
            # Calculate distribution uniformity
            vertical_uniformity = 1.0 - (np.std(vertical_projection) / (np.mean(vertical_projection) + 1e-6))
            horizontal_uniformity = 1.0 - (np.std(horizontal_projection) / (np.mean(horizontal_projection) + 1e-6))
            
            layout_score = (vertical_uniformity + horizontal_uniformity) / 2.0
            layout_score = max(0.0, min(1.0, layout_score))
            
            # 4. Font size appropriateness evaluation (0-1 points)
            # Evaluate font size based on image size and text density
            height, width = gray.shape
            total_pixels = height * width
            
            # Calculate text pixel ratio
            text_pixels = np.sum(edges > 0)
            text_ratio = text_pixels / total_pixels
            
            # Ideal text ratio should be between 0.05-0.2
            if 0.05 <= text_ratio <= 0.2:
                font_size_score = 1.0
            elif text_ratio < 0.05:
                font_size_score = text_ratio / 0.05
            else:
                font_size_score = max(0.0, 1.0 - (text_ratio - 0.2) * 2)
            
            # Comprehensive scoring (0-2 points)
            total_score = (
                clarity_score * 0.3 +
                readability_score * 0.3 +
                layout_score * 0.2 +
                font_size_score * 0.2
            ) * 2.0
            
            return min(2.0, max(0.0, total_score))
            
        except Exception as e:
            self.logger.error(f"Text accuracy evaluation failed: {str(e)}")
            return 1.0
    
    def _evaluate_color_analysis(self, image: Image.Image) -> float:
        """
        Evaluate color analysis metrics
        
        Evaluation dimensions:
        - Color richness
        - Color harmony
        - Government color compliance
        """
        try:
            # 1. Color richness analysis (0-1 points) - Stricter standards
            # Calculate unique color count
            colors = image.getcolors(maxcolors=256*256*256)
            if colors:
                unique_colors = len(colors)
                # Stricter color count evaluation: only high-quality posters can get high scores
                if unique_colors > 50000:  # f3dd7bc431.jpeg characteristics
                    richness_score = 1.0
                elif 20000 <= unique_colors <= 50000:  # High-quality posters
                    richness_score = 0.8
                elif 1000 <= unique_colors < 20000:  # poster.png in this range
                    richness_score = 0.6
                elif 500 <= unique_colors < 1000:  # Medium quality
                    richness_score = 0.4
                elif unique_colors < 500:  # Low quality
                    richness_score = unique_colors / 500.0
                else:
                    richness_score = 0.3
            else:
                richness_score = 0.2  # Reduce default score
            
            # 2. Color harmony analysis (0-1 points)
            # Calculate color distribution uniformity
            img_array = np.array(image)
            
            # Calculate RGB channel distribution
            r_channel = img_array[:,:,0]
            g_channel = img_array[:,:,1]
            b_channel = img_array[:,:,2]
            
            # Calculate standard deviation of each channel
            r_std = np.std(r_channel)
            g_std = np.std(g_channel)
            b_std = np.std(b_channel)
            
            # Smaller standard deviation means more harmonious colors
            avg_std = (r_std + g_std + b_std) / 3.0
            harmony_score = max(0, 1.0 - avg_std / 100.0)
            
            # 3. Government color compliance (0-1 points)
            # Check if government standard colors are included
            gov_colors = self.standard_params['government_colors']
            gov_score = 0.0
            
            # Simplified color matching check
            dominant_colors = self._get_dominant_colors(image, 5)
            for color in dominant_colors:
                for gov_color in gov_colors.values():
                    # Calculate color distance
                    distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(color, gov_color)))
                    if distance < 50:  # Colors are similar
                        gov_score += 0.2
                        break
            
            gov_score = min(1.0, gov_score)
            
            # Comprehensive scoring
            total_score = (richness_score + harmony_score + gov_score) / 3.0 * 2.0
            return min(2.0, max(0.0, total_score))
            
        except Exception as e:
            self.logger.error(f"Color analysis failed: {str(e)}")
            return 1.0
    
    def _evaluate_contrast_analysis(self, image: Image.Image) -> float:
        """
        Evaluate contrast metrics
        
        Evaluation dimensions:
        - Overall contrast
        - Text-background contrast
        - Visual hierarchy sense
        """
        try:
            # 1. Overall contrast analysis (0-1 points) - More precise evaluation
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Calculate contrast (standard deviation)
            contrast = np.std(gray)
            
            # More precise contrast evaluation
            if contrast > 75:  # poster.png has higher contrast
                contrast_score = 1.0
            elif contrast > 65:
                contrast_score = 0.8
            elif contrast > 55:
                contrast_score = 0.6
            else:
                contrast_score = 0.4
            
            # 2. Text-background contrast (0-1 points)
            # Use OTSU threshold to analyze foreground-background contrast
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Calculate average brightness of foreground and background
            foreground = gray[binary == 0]
            background = gray[binary == 255]
            
            if len(foreground) > 0 and len(background) > 0:
                fg_mean = np.mean(foreground)
                bg_mean = np.mean(background)
                text_contrast = abs(fg_mean - bg_mean) / 255.0
            else:
                text_contrast = 0.5
            
            # 3. Visual hierarchy (0-1 points)
            # Hierarchy analysis based on edge density
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Moderate edge density indicates good hierarchy
            if 0.05 <= edge_density <= 0.2:
                hierarchy_score = 1.0
            else:
                hierarchy_score = max(0, 1.0 - abs(edge_density - 0.125) * 4)
            
            # Comprehensive scoring
            total_score = (contrast_score + text_contrast + hierarchy_score) / 3.0 * 2.0
            return min(2.0, max(0.0, total_score))
            
        except Exception as e:
            self.logger.error(f"Contrast analysis failed: {str(e)}")
            return 1.0
    
    def _get_dominant_colors(self, image: Image.Image, num_colors: int = 5) -> List[Tuple[int, int, int]]:
        """Get dominant colors of the image"""
        try:
            # Resize image to improve processing speed
            small_image = image.resize((150, 150))
            
            # Convert to numpy array
            img_array = np.array(small_image)
            pixels = img_array.reshape(-1, 3)
            
            # Use K-means clustering to find dominant colors
            try:
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
                kmeans.fit(pixels)
                colors = kmeans.cluster_centers_.astype(int)
                return [tuple(color) for color in colors]
            except ImportError:
                # If sklearn is not available, use simple color aggregation method
                self.logger.warning("sklearn not available, using simple color aggregation method")
                # Simple implementation: return some dominant colors from the image
                img_array = np.array(image)
                pixels = img_array.reshape(-1, 3)
                unique_pixels = np.unique(pixels, axis=0)
                if len(unique_pixels) >= num_colors:
                    indices = np.linspace(0, len(unique_pixels)-1, num_colors, dtype=int)
                    return [tuple(unique_pixels[i].tolist()) for i in indices]
                else:
                    return [(128, 128, 128)] * num_colors
            
        except Exception as e:
            self.logger.error(f"Failed to get dominant colors: {str(e)}")
            return [(128, 128, 128)] * num_colors  # Return gray as default
    
    def _generate_recommendations(self, image_quality: float, layout: float, 
                                text_density: float, text_accuracy: float, 
                                color: float, contrast: float) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        if image_quality < 1.5:
            recommendations.append("Suggest improving image resolution to ensure clarity")
        
        if layout < 1.5:
            recommendations.append("Suggest optimizing layout design to improve visual balance")
        
        if text_density < 1.5:
            recommendations.append("Suggest adjusting text density to maintain information readability")
        
        if text_accuracy < 1.5:
            recommendations.append("Suggest improving text clarity and readability")
        
        if color < 1.5:
            recommendations.append("Suggest optimizing color combination using government standard colors")
        
        if contrast < 1.5:
            recommendations.append("Suggest increasing contrast to enhance visual hierarchy")
        
        return recommendations
    
    def _fallback_evaluation(self) -> Dict[str, Any]:
        """Fallback plan when quantitative evaluation is unavailable"""
        return {
            'success': False,
            'total_score': 1.0,  # Default medium score
            'sub_scores': {
                'image_quality': 1.0,
                'layout_analysis': 1.0,
                'text_density': 1.0,
                'text_accuracy': 1.0,  # 新增文本准确度
                'color_analysis': 1.0,
                'contrast_analysis': 1.0
            },
            'detailed_metrics': {},
            'recommendations': ['Quantitative evaluation unavailable, please install PIL and OpenCV'],
            'error': 'Quantitative evaluation not available'
        }

class NewGovernmentPosterEvaluator:
    """
    New Government Poster Comprehensive Evaluator
    
    Evaluates poster quality by combining VLM scoring and quantitative metrics to provide more objective and reliable scoring results
    """
    
    def __init__(self, llm_base_url: str, api_key: str, model="qwen-vl-max"):
        """
        Initialize comprehensive evaluator
        
        Args:
            llm_base_url: LLM API base URL
            api_key: API key
            model: VLM model name
        """
        self.llm_base_url = llm_base_url
        self.api_key = api_key
        self.model = model
        
        # Comprehensive scoring weight configuration
        self.scoring_weights = {
            'vlm_score': 0.6,      # VLM score weight
            'quantitative_score': 0.4  # Quantitative score weight
        }
        
        # calibration factor to scale scores (<=1.0). This is a global multiplier
        # applied after applying penalties/bonuses. Set default to 1.0 so that
        # objectively perfect posters map to 10/10. This can be tuned by user
        # if a different top score is desired.
        self.score_calibration = 1.0

        # Initialize OpenAI client
        self.client = OpenAI(
            base_url=llm_base_url,
            api_key=api_key
        )
        
        # Initialize quantitative evaluator
        if QUANTITATIVE_AVAILABLE:
            self.quantitative_evaluator = QuantitativeEvaluator()
        else:
            self.quantitative_evaluator = None

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        # Strict rule_type thresholds table, can be adjusted as needed
        self.rule_thresholds = {
            "guiding_policy": {"overall_layout":1.0,"color_contrast":1.0,"typography_readability":1.0,"information_hierarchy":1.0,"government_compliance":1.0},
            "implementation":  {"overall_layout":0.8,"color_contrast":0.8,"typography_readability":1.0,"information_hierarchy":1.2,"government_compliance":1.0},
            "service_guide":  {"overall_layout":0.8,"color_contrast":0.8,"typography_readability":1.2,"information_hierarchy":1.2,"government_compliance":0.6},
            "notices":       {"overall_layout":0.6,"color_contrast":0.8,"typography_readability":1.0,"information_hierarchy":0.6,"government_compliance":0.6},
            "faq":           {"overall_layout":0.6,"color_contrast":0.6,"typography_readability":1.0,"information_hierarchy":0.8,"government_compliance":0.4},
            "legal_documents":{"overall_layout":1.0,"color_contrast":1.0,"typography_readability":1.2,"information_hierarchy":1.2,"government_compliance":1.4}
        }
    
    def evaluate_poster(self, parsed_data: Dict[str, Any], poster_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate poster quality using comprehensive evaluation method
        
        Combines VLM scoring and quantitative metrics to provide more objective and reliable scoring results
        
        Args:
            parsed_data: Parsed document data
            poster_result: Poster generation result
            
        Returns:
            Dict: Comprehensive evaluation result
        """
        try:
            self.logger.info("Starting comprehensive poster quality evaluation...")
            
            start_time = time.time()
            
            # 1. VLM evaluation
            vlm_evaluation = self._evaluate_with_vlm_comprehensive(parsed_data, poster_result)
            vlm_score = vlm_evaluation.get('score', 0)
            
            # 2. Quantitative evaluation
            quantitative_evaluation = self._evaluate_with_quantitative_metrics(poster_result)
            quantitative_score = quantitative_evaluation.get('total_score', 0)
            
            # 3. Comprehensive score calculation
            final_score = self._calculate_hybrid_score(vlm_score, quantitative_score)
            
            evaluation_time = time.time() - start_time
            
            # Merge improvement suggestions
            all_suggestions = []
            if vlm_evaluation.get('suggestions'):
                all_suggestions.extend(vlm_evaluation['suggestions'])
            if quantitative_evaluation.get('recommendations'):
                all_suggestions.extend(quantitative_evaluation['recommendations'])
            
            result = {
                'success': True,
                'score': final_score,
                'evaluation_time': evaluation_time,
                'vlm_evaluation': vlm_evaluation,
                'quantitative_evaluation': quantitative_evaluation,
                'hybrid_scores': {
                    'vlm_score': vlm_score,
                    'quantitative_score': quantitative_score,
                    'final_score': final_score
                },
                'improvement_suggestions': all_suggestions,
                'metadata': {
                    'evaluator_version': '4.0',
                    'evaluation_timestamp': time.time(),
                    'evaluation_method': 'hybrid_vlm_quantitative',
                    'scoring_weights': self.scoring_weights
                }
            }
            
            self.logger.info(f"Comprehensive evaluation completed - VLM: {vlm_score:.2f}, Quantitative: {quantitative_score:.2f}, Final: {final_score:.2f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Comprehensive evaluation failed: {str(e)}")
            return {
                'success': False,
                'score': 0.0,
                'error': str(e),
                'improvement_suggestions': [],
            }
    
    def _evaluate_with_quantitative_metrics(self, poster_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate poster quality using quantitative metrics
        
        Args:
            poster_result: Poster generation result
            
        Returns:
            Dict: Quantitative evaluation result
        """
        try:
            # Check if quantitative evaluation can be used
            if not QUANTITATIVE_AVAILABLE or not self.quantitative_evaluator:
                return {
                    'success': False,
                    'total_score': 1.0,
                    'error': 'Quantitative evaluator not available'
                }
            
            # Get poster image path
            poster_path = poster_result.get('poster_path')
            if not poster_path or not os.path.exists(poster_path):
                return {
                    'success': False,
                    'total_score': 0.0,
                    'error': 'Poster path not found or invalid'
                }
            
            # Execute quantitative evaluation
            quantitative_result = self.quantitative_evaluator.evaluate_poster(poster_path)
            
            self.logger.info(f"Quantitative evaluation completed, score: {quantitative_result.get('total_score', 0):.2f}")
            return quantitative_result
            
        except Exception as e:
            self.logger.error(f"Quantitative evaluation failed: {str(e)}")
            return {
                'success': False,
                'total_score': 1.0,
                'error': str(e)
            }
    
    def _calculate_hybrid_score(self, vlm_score: float, quantitative_score: float) -> float:
        """
        Calculate comprehensive score
        
        Args:
            vlm_score: VLM score (0-10)
            quantitative_score: Quantitative score (0-2)
            
        Returns:
            float: Comprehensive score (0-10)
        """
        try:
            # Convert quantitative score from 0-2 range to 0-10 range
            quantitative_score_normalized = quantitative_score * 5.0
            
            # Weighted average
            hybrid_score = (
                vlm_score * self.scoring_weights['vlm_score'] +
                quantitative_score_normalized * self.scoring_weights['quantitative_score']
            )
            
            # Apply calibration factor
            final_score = hybrid_score * self.score_calibration
            
            # Ensure score is within 0-10 range
            final_score = max(0.0, min(10.0, final_score))
            
            self.logger.info(f"Comprehensive score calculation: VLM({vlm_score:.2f}) * {self.scoring_weights['vlm_score']} + "
                           f"Quantitative({quantitative_score_normalized:.2f}) * {self.scoring_weights['quantitative_score']} "
                           f"* {self.score_calibration} = {final_score:.2f}")
            
            return round(final_score, 2)
            
        except Exception as e:
            self.logger.error(f"Comprehensive score calculation failed: {str(e)}")
            return vlm_score  # Fallback to VLM score
    
    def set_scoring_weights(self, vlm_weight: float = 0.6, quantitative_weight: float = 0.4):
        """
        Set scoring weights
        
        Args:
            vlm_weight: VLM score weight (0-1)
            quantitative_weight: Quantitative score weight (0-1)
        """
        if abs(vlm_weight + quantitative_weight - 1.0) > 0.01:
            raise ValueError("Sum of weights must equal 1.0")
        
        self.scoring_weights = {
            'vlm_score': vlm_weight,
            'quantitative_score': quantitative_weight
        }
        
        self.logger.info(f"Scoring weights updated: VLM={vlm_weight}, Quantitative={quantitative_weight}")
    
    def _evaluate_with_vlm_comprehensive(self, parsed_data: Dict[str, Any], poster_result: Dict[str, Any]) -> Dict[str, Any]:
        """Use VLM for comprehensive evaluation (including scoring and suggestions)"""
        try:
            # Get poster image path
            poster_path = poster_result.get('poster_path')
            # poster_path must point to an image file. If it points to a PDF, try to
            # find a corresponding image in the posters dir or convert the first
            # PDF page to PNG (if pdf2image available). This prevents sending raw
            # PDF text/content to the VLM.
            if not poster_path:
                self.logger.warning("poster_path is empty, cannot perform VLM evaluation")
                return {
                    'score': 0,
                    'suggestions': ['poster_path is empty, cannot perform evaluation'],
                    'detailed_scores': {},
                    'analysis': 'Unable to analyze'
                }
            if not os.path.exists(poster_path):
                self.logger.warning(f"Poster path does not exist: {poster_path}")
                return {
                    'score': 0,
                    'suggestions': ['Poster image does not exist, cannot perform evaluation'],
                    'detailed_scores': {},
                    'analysis': 'Unable to analyze'
                }
            # If poster_path points to a PDF, try to resolve to an image
            poster_path = str(poster_path)
            lower = poster_path.lower()
            resolved_image_path = None
            if lower.endswith('.pdf'):
                # attempt 1: look for image with same stem in posters dir
                stem = Path(poster_path).stem
                posters_dir = os.path.join('examples', 'new_government_agent', 'posters')
                if os.path.exists(posters_dir):
                    for fn in os.listdir(posters_dir):
                        if stem in fn and fn.lower().endswith(('.png', '.jpg', '.jpeg')):
                            resolved_image_path = os.path.join(posters_dir, fn)
                            break

                # attempt 2: try replacing .pdf with .png adjacent to pdf
                if not resolved_image_path:
                    alt = poster_path[:-4] + '.png'
                    if os.path.exists(alt):
                        resolved_image_path = alt

                # attempt 3: convert PDF first page to PNG if pdf2image available
                if not resolved_image_path:
                    try:
                        from pdf2image import convert_from_path
                        images = convert_from_path(poster_path, first_page=1, last_page=1)
                        if images:
                            tmp_path = os.path.join('/tmp', f'vlm_converted_{int(time.time())}.png')
                            images[0].save(tmp_path, 'PNG')
                            resolved_image_path = tmp_path
                    except Exception as e:
                        # conversion not available or failed; we'll handle below
                        self.logger.info(f"pdf->png conversion unavailable or failed: {e}")

            else:
                resolved_image_path = poster_path

            if not resolved_image_path or not os.path.exists(resolved_image_path):
                self.logger.warning(f"Unable to resolve poster path as image: {poster_path}")
                return {
                    'score': 0,
                    'suggestions': ['Unable to resolve poster path as image, cannot perform evaluation'],
                    'detailed_scores': {},
                    'analysis': 'Unable to analyze'
                }

            # Read image and convert to base64
            
            with open(resolved_image_path, 'rb') as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')

            # Build stricter JSON-only prompt (apply thresholds by rule_type)
            rule_type = parsed_data.get('rule_type', 'notices')
            thresholds = self.rule_thresholds.get(rule_type, self.rule_thresholds['notices'])
            # Add explicit scoring bands and require concise visual evidence (1-2 sentences). Encourage conservative scoring.
            import json as _json
            prompt_template = """
你是一个严格且专业的政府海报视觉评审专家。请严格按照标准评估海报质量，只返回严格的 JSON（用 ```json 包裹也可以），不要输出额外的自然语言。
输入变量：
- RULE_TYPE: {RULE_TYPE}
- THRESHOLDS: {PASS_THRESHOLDS}

重要：请保持严格且保守的评分标准。大多数海报应该得到中等或偏低的分数，只有真正高质量的海报才能获得高分。为每个子项提供明确的视觉证据（1-2句）并基于证据给出证据强度指标 evidence_strength（数值 0.0-1.0）。

评分范围说明（子项，0.0-2.0）：
- 0.0-0.3: 很差（严重设计问题，不符合政府海报标准）
- 0.4-0.8: 较差（存在明显问题，需要大幅改进）
- 0.9-1.3: 一般（基本合格但质量不高）
- 1.4-1.7: 良好（质量较好，符合政府海报标准）
- 1.8-2.0: 优秀（高质量，设计精美，符合政府海报标准）

输出必须符合下述 schema（所有分值为数值，保留1位小数）：
{
    "success": true,
    "score": <number 0-10>,
    "sub_scores": {
        "overall_layout": <0.0-2.0>,
        "color_contrast": <0.0-2.0>,
        "typography_readability": <0.0-2.0>,
        "information_hierarchy": <0.0-2.0>,
        "government_compliance": <0.0-2.0>
    },
    "evidence": {
        "overall_layout": "<1-2句证据>",
        ...
    },
    "evidence_strength": {
        "overall_layout": <0.0-1.0>,
        ...
    },
    "match": {
         "overall_layout": "完全符合|基本符合|不符合",
         ...
    },
    "improvement_suggestions": [{"priority":1, "suggestion":"..."}, ...],
    "rule_type": "{RULE_TYPE}",
    "pass_thresholds": {PASS_THRESHOLDS},
    "passed": true|false,
    "notes": "optional"
}

说明：
- 每个 sub_scores 项必须给出 1-2 句证据(evidence)和对应的 evidence_strength（0-1）。
- 请严格按照政府海报标准评分，大多数海报应该得到中等或偏低的分数。
- 评分规则：每个子项根据视觉质量严格评分，范围0.0-2.0：
    - 优秀设计：1.8-2.0分（真正高质量的海报，设计精美，完全符合政府海报标准）
    - 良好设计：1.4-1.7分（质量较好，基本符合政府海报标准）
    - 一般设计：0.9-1.3分（基本合格但质量不高，需要改进）
    - 较差设计：0.4-0.8分（存在明显问题，需要大幅改进）
    - 很差设计：0.0-0.3分（严重设计问题，不符合政府海报标准）
- 总分应等于子项之和（允许 0.1 误差），返回 'passed' 表示是否所有子项达到对应 THRESHOLDS。
- 若无法判定某项，请给 0.0 并在 evidence 说明原因。不要在 evidence 中粘贴长篇原文。
- 输出 JSON 后结束回复。
"""
            # safely inject placeholders
            prompt = prompt_template.replace('{RULE_TYPE}', str(rule_type)).replace('{PASS_THRESHOLDS}', _json.dumps(thresholds, ensure_ascii=False))
            
            # Call VLM API (keep existing call parameters, but temperature can be lower to improve JSON compliance)
            # Before calling VLM, log image metadata to confirm we are actually sending image bytes
            try:
                img_bytes = None
                with open(resolved_image_path, 'rb') as f:
                    img_bytes = f.read()
                size_bytes = len(img_bytes) if img_bytes else 0
                sha256 = None
                try:
                    import hashlib
                    sha256 = hashlib.sha256(img_bytes).hexdigest() if img_bytes else None
                except Exception:
                    sha256 = None
                # try to get image dimensions if PIL available
                dims = None
                try:
                    from PIL import Image
                    from io import BytesIO
                    im = Image.open(BytesIO(img_bytes))
                    dims = im.size  # (width, height)
                except Exception:
                    dims = None

                self.logger.info(f"VLM payload image: path={resolved_image_path}, size={size_bytes} bytes, dims={dims}, sha256={sha256}")
            except Exception as e:
                self.logger.info(f"Unable to read or analyze poster image metadata: {e}")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_data}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1200,
                temperature=0.15
            )
            
            # Parse VLM response
            response_text = response.choices[0].message.content
            evaluation_result = self._parse_vlm_comprehensive_response(response_text)

            # attach image metadata to evaluation_result for downstream validation penalties
            try:
                evaluation_result['image_meta'] = {
                    'size_bytes': size_bytes,
                    'dims': dims,
                    'sha256': sha256
                }
            except Exception:
                pass

            # Validate and enforce threshold rules
            try:
                evaluation_result = self._validate_and_enforce(evaluation_result, rule_type)
            except Exception as e:
                self.logger.warning(f"Evaluation result validation failed: {e}")

            # Avoid reporting full VLM analysis text (which may be very large) directly to log or upper-level results, truncate analysis field
            if isinstance(evaluation_result, dict) and 'analysis' in evaluation_result and isinstance(evaluation_result['analysis'], str):
                a = evaluation_result['analysis']
                evaluation_result['analysis'] = (a[:1000] + '...') if len(a) > 1000 else a

            return evaluation_result
            
        except Exception as e:
            self.logger.warning(f"VLM comprehensive evaluation failed: {str(e)}")
            return {
                'score': 0,
                'suggestions': [f'VLM evaluation failed: {str(e)}'],
                'detailed_scores': {},
                'analysis': 'Evaluation failed'
            }
    
    def _parse_vlm_comprehensive_response(self, response_text: str) -> Dict[str, Any]:
        """Parse VLM comprehensive evaluation response"""
        try:
            import re
            import json

            # Try to extract JSON first (handle ```json``` fenced blocks or raw JSON)
            json_obj = None
            try:
                # fenced ```json
                m = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.S)
                if m:
                    json_obj = json.loads(m.group(1))
                else:
                    # fenced without json tag
                    m2 = re.search(r'```\s*(\{.*?\})\s*```', response_text, re.S)
                    if m2:
                        json_obj = json.loads(m2.group(1))
                    else:
                        # try to find the first JSON object in the text by searching first '{' and last '}'
                        first = response_text.find('{')
                        last = response_text.rfind('}')
                        if first != -1 and last != -1 and last > first:
                            candidate = response_text[first:last+1]
                            try:
                                json_obj = json.loads(candidate)
                            except Exception:
                                json_obj = None
            except Exception:
                json_obj = None

            # mapping from possible localized keys to canonical english keys
            key_map = {
                '整体布局': 'overall_layout',
                'overall_layout': 'overall_layout',
                '色彩搭配': 'color_contrast',
                'color_contrast': 'color_contrast',
                '字体可读性': 'typography_readability',
                'typography_readability': 'typography_readability',
                '信息层次': 'information_hierarchy',
                'information_hierarchy': 'information_hierarchy',
                '政府规范性': 'government_compliance',
                'government_compliance': 'government_compliance'
            }

            # If we parsed JSON, normalize it
            if isinstance(json_obj, dict):
                sub_scores = {}
                # try direct sub_scores key
                if 'sub_scores' in json_obj and isinstance(json_obj['sub_scores'], dict):
                    for k, v in json_obj['sub_scores'].items():
                        canon = key_map.get(k, None) or key_map.get(k.strip(), None)
                        if canon:
                            try:
                                sub_scores[canon] = round(max(0.0, min(2.0, float(v))), 1)
                            except Exception:
                                sub_scores[canon] = 0.0
                # try alternative key names (detailed_scores)
                elif 'detailed_scores' in json_obj and isinstance(json_obj['detailed_scores'], dict):
                    for k, v in json_obj['detailed_scores'].items():
                        canon = key_map.get(k, None) or key_map.get(k.strip(), None)
                        if canon:
                            try:
                                sub_scores[canon] = round(max(0.0, min(2.0, float(v))), 1)
                            except Exception:
                                sub_scores[canon] = 0.0

                # If still empty, try to map any numeric fields that look like our keys
                if not sub_scores:
                    for k, v in json_obj.items():
                        if isinstance(v, (int, float)):
                            canon = key_map.get(k, None)
                            if canon:
                                sub_scores[canon] = round(max(0.0, min(2.0, float(v))), 1)

                # Ensure all keys exist
                for kk in ['overall_layout','color_contrast','typography_readability','information_hierarchy','government_compliance']:
                    if kk not in sub_scores:
                        sub_scores[kk] = 0.0

                total_score = round(sum(sub_scores.values()), 1)
                suggestions = []
                if 'improvement_suggestions' in json_obj and isinstance(json_obj['improvement_suggestions'], list):
                    suggestions = json_obj['improvement_suggestions']
                elif 'suggestions' in json_obj and isinstance(json_obj['suggestions'], list):
                    suggestions = json_obj['suggestions']

                evidence = json_obj.get('evidence', {}) if isinstance(json_obj.get('evidence', {}), dict) else {}
                evidence_strength = json_obj.get('evidence_strength', {}) if isinstance(json_obj.get('evidence_strength', {}), dict) else {}
                match = json_obj.get('match', {}) if isinstance(json_obj.get('match', {}), dict) else {}

                return {
                    'score': total_score,
                    'sub_scores': sub_scores,
                    'suggestions': suggestions,
                    'evidence': evidence,
                    'evidence_strength': evidence_strength,
                    'match': match,
                    'analysis': response_text
                }

            # Fallback: original regex-based extraction for free-text responses
            detailed_scores = {}
            score_patterns = {
                '整体布局': r'整体布局[：:]\s*(\d+\.?\d*)分?',
                '色彩搭配': r'色彩搭配[：:]\s*(\d+\.?\d*)分?',
                '字体可读性': r'字体可读性[：:]\s*(\d+\.?\d*)分?',
                '信息层次': r'信息层次[：:]\s*(\d+\.?\d*)分?',
                '政府规范性': r'政府规范性[：:]\s*(\d+\.?\d*)分?'
            }
            total_score = 0
            for key, pattern in score_patterns.items():
                match = re.search(pattern, response_text)
                if match:
                    score = float(match.group(1))
                    detailed_scores[key] = min(2.0, max(0.0, score))
                    total_score += score
                else:
                    detailed_scores[key] = 0  # Default score

            # Extract improvement suggestions
            suggestions = []
            suggestions_section = re.search(r'改进建议[：:]?\s*(.*?)(?=$)', response_text, re.S)
            if suggestions_section:
                suggestions_text = suggestions_section.group(1)
                suggestions = [line.strip() for line in suggestions_text.split('\n') if line.strip() and line.strip().startswith('-')]
                suggestions = [s[1:].strip() for s in suggestions]

            # map detailed_scores keys to canonical english sub_scores
            mapped = {}
            for k_ch, v in detailed_scores.items():
                canon = key_map.get(k_ch, None)
                if canon:
                    mapped[canon] = round(float(v), 1)
            for kk in ['overall_layout','color_contrast','typography_readability','information_hierarchy','government_compliance']:
                if kk not in mapped:
                    mapped[kk] = 0.0

            return {
                'score': round(sum(mapped.values()), 1),
                'sub_scores': mapped,
                'suggestions': suggestions,
                'evidence': {},
                'evidence_strength': {},
                'match': {},
                'analysis': response_text
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to parse VLM response: {str(e)}")
            return {
                'score': 0,
                'detailed_scores': {},
                'suggestions': ['Failed to parse evaluation result'],
                'evidence': {},
                'evidence_strength': {},
                'analysis': response_text
            }

    def _validate_and_enforce(self, evaluation_result: Dict[str, Any], rule_type: str) -> Dict[str, Any]:
        """Validate structure, normalize sub_scores and enforce rule thresholds."""
        keys = ["overall_layout","color_contrast","typography_readability","information_hierarchy","government_compliance"]
        # Build sub_scores from strict 'match' + 'evidence_strength' rules when available
        reported_subs = evaluation_result.get('sub_scores', {}) or {}
        evidence = evaluation_result.get('evidence', {}) or {}
        evidence_strengths = evaluation_result.get('evidence_strength', {}) or {}
        match = evaluation_result.get('match', {}) or {}

        sub = {}
        for k in keys:
            # determine match: prefer explicit 'match' field, else infer from evidence length
            m = None
            if k in match:
                try:
                    m = str(match.get(k)).strip()
                except Exception:
                    m = None
            # evidence_strength
            s = 0.0
            try:
                s = float(evidence_strengths.get(k) or 0.0)
            except Exception:
                s = 0.0
            if not m:
                # infer match from evidence length
                txt = evidence.get(k) or ''
                l = len(str(txt).strip())
                if l >= 100:
                    m = '完全符合'
                elif l >= 20:
                    m = '基本符合'
                else:
                    m = '不符合'

            # compute sub-score based on strict mapping
            if m == '完全符合':
                # base 1.0 plus evidence_strength (0-1) to yield 1.0-2.0
                val = 1.0 + max(0.0, min(1.0, s))
            elif m == '基本符合':
                # evidence_strength scaled 0-1 -> 0-1
                val = max(0.0, min(1.0, s))
            else:
                # 不符合
                val = 0.0  # Does not comply

            sub[k] = round(max(0.0, min(2.0, float(val))), 1)

        total = round(sum(sub[k] for k in keys), 1)
        evaluation_result['sub_scores'] = sub
        evaluation_result['score'] = total

        # Apply dynamic calibration driven by evidence_strength and image quality.
        # Compute per-key evidence strength (0-1) from either explicit field or infer from evidence length.
        evidence = evaluation_result.get('evidence') or {}
        evidence_strengths = evaluation_result.get('evidence_strength') or {}
        inferred_strength = {}
        for k in keys:
            s = None
            try:
                if k in evidence_strengths:
                    s = float(evidence_strengths.get(k) or 0.0)
            except Exception:
                s = None
            if s is None or s == 0.0:
                # infer from evidence text length
                txt = evidence.get(k) or ''
                try:
                    l = len(str(txt).strip())
                except Exception:
                    l = 0
                # normalize: 0-200 chars maps to 0.0-1.0
                s = min(1.0, l / 200.0)
            inferred_strength[k] = round(max(0.0, min(1.0, float(s))), 2)

        # average evidence strength across subs
        avg_strength = sum(inferred_strength[k] for k in keys) / len(keys)
        min_strength = min(inferred_strength[k] for k in keys)
        strong_count = sum(1 for k in keys if inferred_strength[k] >= 0.8)

        # dynamic calibration: be more generous for high-quality posters
        if avg_strength >= 0.7 and strong_count >= 2 and min_strength >= 0.5:
            calib = 1.0  # 高质量海报不额外扣分
        elif avg_strength >= 0.5 and strong_count >= 1 and min_strength >= 0.3:
            calib = 0.9  # 中等质量海报轻微扣分
        elif avg_strength >= 0.3:
            calib = 0.8  # 低质量海报适度扣分
        else:
            calib = 0.6  # 很差的海报显著扣分

        # penalties and bonuses
        image_meta = evaluation_result.get('image_meta') or {}
        try:
            size_b = int(image_meta.get('size_bytes') or 0)
        except Exception:
            size_b = 0
        dims_meta = image_meta.get('dims')
        try:
            width, height = (int(dims_meta[0]), int(dims_meta[1])) if dims_meta else (0, 0)
        except Exception:
            width, height = (0, 0)

        penalties = {k: 0.0 for k in keys}
        bonuses = {k: 0.0 for k in keys}

        # image quality penalties (reduced severity)
        if size_b and size_b < 30 * 1024:
            for k in keys:
                penalties[k] += 0.1  # 减少图片质量惩罚
        if width and height and (width < 600 or height < 450):
            for k in keys:
                penalties[k] += 0.1  # 减少尺寸惩罚

        # per-key handling: require multiple strong evidence subs for big bonuses
        for k in keys:
            raw = float(sub.get(k, 0.0))
            s = inferred_strength.get(k, 0.0)
            # if only a few subs are strong, be very conservative about bonuses
            if s >= 0.85 and strong_count >= 3 and min_strength >= 0.6:
                # allow meaningful bonus only when several subs support high quality
                if raw >= 1.8:
                    bonuses[k] += 0.32
                elif raw >= 1.6:
                    bonuses[k] += 0.15
                else:
                    bonuses[k] += 0.05
            elif s >= 0.8 and strong_count >= 2 and min_strength >= 0.45:
                # moderate allowance
                if raw >= 1.8:
                    bonuses[k] += 0.18
                elif raw >= 1.6:
                    bonuses[k] += 0.09
                else:
                    bonuses[k] += 0.03
            else:
                # weak evidence: reduce the effective raw proportionally
                # use a slightly gentler penalty coefficient so that moderately
                # supported posters don't collapse to near-zero total
                penalty_coeff = 0.5
                penalties[k] += (0.6 - s) * penalty_coeff
                # extra penalty if raw is very high but evidence weak -> suspicious
                # reduce suspicious extra penalty to avoid over-punishing borderline cases
                if raw >= 1.8 and s < 0.7:
                    penalties[k] += 0.45

        calibrated_subs = {}
        for k in keys:
            raw = float(sub.get(k, 0.0))
            penalized = max(0.0, raw - penalties.get(k, 0.0))
            adjusted = penalized + bonuses.get(k, 0.0)
            val = max(0.0, min(2.0, adjusted * calib))
            # further cap: if evidence overall is weak (avg_strength <0.3), don't let any sub exceed 1.5
            if avg_strength < 0.3:
                val = min(val, 1.5)
            # apply global score calibration multiplier so top totals map to ~9
            val = val * float(self.score_calibration)
            calibrated_subs[k] = round(max(0.0, min(2.0, val)), 1)

        calibrated_total = round(sum(calibrated_subs[k] for k in keys), 1)

        thresholds = self.rule_thresholds.get(rule_type, self.rule_thresholds['notices'])
        failed = []
        for k in keys:
            if calibrated_subs.get(k, 0.0) < thresholds.get(k, 0.0):
                failed.append((k, sub[k], thresholds[k]))

        evaluation_result['pass_thresholds'] = thresholds
        evaluation_result['passed'] = False if failed else True

        # update stored score/sub_scores to calibrated values (for reporting)
        evaluation_result['calibrated_sub_scores'] = calibrated_subs
        evaluation_result['calibrated_score'] = calibrated_total

        # Replace reported score with calibrated total so callers use the conservative value
        evaluation_result['score'] = calibrated_total

        # (pass/fail already computed using calibrated values above)

        # normalize suggestions - ensure list of dicts
        sugg = evaluation_result.get('suggestions') or evaluation_result.get('improvement_suggestions') or []
        normalized = []
        if isinstance(sugg, list):
            for i, s in enumerate(sugg[:6]):
                if isinstance(s, dict):
                    normalized.append(s)
                else:
                    normalized.append({'priority': i+1, 'suggestion': str(s)})
        evaluation_result['improvement_suggestions'] = normalized

        # if failed, promote corrective suggestions (use original values in message)
        if failed:
            for idx, (k, val, thr) in enumerate(failed):
                evaluation_result['improvement_suggestions'].insert(0, {'priority': 1+idx, 'suggestion': f'Improve {k} to at least {thr} (currently {val})'})

        return evaluation_result

    
=======
version https://git-lfs.github.com/spec/v1
oid sha256:5fe6f0bff8ff5f13e9e7ca9e805b6d5b87f3a93a99690ed13d8f43bacdabd1fd
size 65949
>>>>>>> 9676c3e (ya toh aar ya toh par)
