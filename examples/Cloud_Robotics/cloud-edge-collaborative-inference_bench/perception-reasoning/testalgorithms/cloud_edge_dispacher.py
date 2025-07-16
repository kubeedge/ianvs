# Copyright 2024 The KubeEdge Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import, division, print_function

import os
import abc
import cv2
import numpy as np

from sedna.common.class_factory import ClassType, ClassFactory
from core.common.log import LOGGER
from edge_model import EdgeModel

__all__ = ['EdgeCLoudDispatcher']

# Global configuration for Cityscapes dataset
CITYSCAPES_CLASSES = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                      'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
                      'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
                      'bicycle']
CITYSCAPES_COLORMAP = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                       [190, 153, 153], [153, 153, 153], [
                           250, 170, 30], [220, 220, 0],
                       [107, 142, 35], [152, 251, 152], [
                           70, 130, 180], [220, 20, 60],
                       [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
                       [0, 80, 100], [0, 0, 230], [119, 11, 32]]
CRITICAL_CLASSES = ['person', 'rider', 'car', 'truck',
                  'bus', 'train', 'motorcycle', 'bicycle']


class BaseFilter(metaclass=abc.ABCMeta):
    """Abstract base class for edge-cloud filtering strategies.

    Provides unified interface for hard sample detection in edge computing scenarios.
    """

    def __init__(self, **kwargs):
        """Initialize filter with optional configuration.

        Args:
            **kwargs: Filter-specific parameters
        """
        LOGGER.info(f"USING {self.__class__.__name__}")

    def __call__(self, infer_result=None):
        """Determine if sample requires cloud processing.

        Args:
            infer_result: Prediction results from edge model

        Returns:
            bool: True if sample requires cloud processing

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError

    @classmethod
    def data_check(cls, data):
        """Validate input data range.

        Args:
            data: Input value to check

        Returns:
            bool: True if data in [0,1] range
        """
        return 0 <= float(data) <= 1


@ClassFactory.register(ClassType.HEM, alias="EdgeCloudDispatcher")
class EdgeCLoudDispatcher(BaseFilter, abc.ABC):
    """Dispatcher for edge-cloud computing workflow.

    Makes routing decisions based on segmentation results from edge model.
    """

    def __init__(self,):
        """Initialize with edge model instance."""
        self.edge_model = EdgeModel()

    def __call__(self, input_data):
        """Execute full edge-cloud inference pipeline.

        Args:
            input_data: Input image path or data

        Returns:
            tuple: (edge_result, cloud_result) depending on routing decision

        Example:
            >>> dispatcher = EdgeCloudDispatcher()
            >>> edge_out, cloud_out = dispatcher("test.jpg")
        """
        mask = self.edge_model(input_data)
        risk = self._rule_based_decision(mask)
        LOGGER.info(f"Cloud offloading decision: {risk}")
        os.environ["CLOUD_OFFLOADING_DECISION"] = risk
        if risk in ['low','high', 'medium']:
            # If decision is to send to cloud, perform inference on the cloud model
            return True
        else:
            # Otherwise, return the edge model output
            return False

    def _rule_based_decision(self, mask, class_labels=CITYSCAPES_CLASSES, critical_classes=CRITICAL_CLASSES):
        """Make cloud offloading decision based on segmentation analysis.

        Implements three-level decision logic combining:
        - Safety-critical object detection
        - Scene complexity evaluation
        - Spatial distribution analysis

        Args:
            mask (np.ndarray): 2D segmentation mask with class IDs
            class_labels (list): Ordered list of class names corresponding to mask IDs
            critical_classes (list): Classes requiring high-priority processing

        Returns:
            str: Processing tier - 'high'|'medium'|'low'

        Decision Logic:
            1. 'high' if critical objects present
            2. 'high' if complexity_score > 0.7
            3. 'medium' if 0.4 < complexity_score ≤ 0.7
            4. 'low' otherwise

        Example:
            >>> decision = dispatcher._rule_based_decision(mask)
            >>> print(f"Recommended processing: {decision}")
        """
        # Critical safety objects always need high resolution
        unique_labels = [class_labels[i] for i in np.unique(mask)]
        if any(cls in critical_classes for cls in unique_labels):
            return 'high'

        # Complexity analysis
        object_count = len(np.unique(mask)) - 1  # Exclude background
        edge_density = self._sobel_edge_ratio(mask)
        fragmentation = self._connected_components_count(mask)
        LOGGER.info(f"Object count: {object_count}, Edge density: {edge_density:.3f}, Fragmentation: {fragmentation}")
        complexity_score =  (min(object_count / 15, 1) * 0.4 + 
                             min(edge_density / 0.5, 1) * 0.4 + 
                             min(fragmentation / 20, 1) * 0.2)
        LOGGER.info(f"Complexity score: {complexity_score:.3f}")
        if complexity_score > 0.7:
            return 'high'
        elif complexity_score > 0.4:
            return 'medium'
        else:
            return 'low'

    def _sobel_edge_ratio(self, mask):
        """Calculate edge density using Sobel operator.

        Technical Details:
        1. Applies 3x3 Sobel kernels for gradient detection
        2. Computes magnitude from X/Y gradients
        3. Uses binary threshold to identify edge pixels

        Args:
            mask (np.ndarray): Input segmentation mask (H x W)

        Returns:
            float: Edge density ratio in range [0,1]

        Note:
            Higher values indicate more complex scene boundaries
        """
        # Convert mask to uint8 if not already
        mask_uint8 = mask.astype(np.uint8)
        # Compute Sobel gradients
        sobelx = cv2.Sobel(mask_uint8, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(mask_uint8, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(sobelx**2 + sobely**2)
        # Threshold to get edge pixels
        edge_pixels = (sobel_mag > 0).sum()
        total_pixels = mask.size
        return edge_pixels / total_pixels

    def _connected_components_count(self, mask):
        """Quantify object fragmentation in segmentation results.

        Methodology:
        1. Processes each class separately (excluding background)
        2. Uses 8-connectivity component labeling
        3. Accumulates fragments across all classes

        Args:
            mask (np.ndarray): Input segmentation mask (H x W)

        Returns:
            int: Total number of disconnected object fragments

        Note:
            Higher counts indicate more scattered objects
        """
        # Convert mask to uint8 if not already
        mask_uint8 = mask.astype(np.uint8)
        # Exclude background (assumed to be class 0)
        # For each class (except background), count connected components
        total_components = 0
        for class_id in np.unique(mask_uint8):
            if class_id == 0:
                continue
            class_mask = (mask_uint8 == class_id).astype(np.uint8)
            num_labels, _ = cv2.connectedComponents(class_mask)
            # Subtract 1 to exclude background in this class mask
            total_components += (num_labels - 1)
        return total_components
