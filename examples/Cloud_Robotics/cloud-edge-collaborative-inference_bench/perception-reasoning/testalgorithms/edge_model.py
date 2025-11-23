<<<<<<< HEAD
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
import numpy as np
from core.common.log import LOGGER
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from sedna.common.class_factory import ClassType, ClassFactory


CITYSCAPES_CLASSES = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
               'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
               'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
               'bicycle']

os.environ['BACKEND_TYPE'] = 'TORCH'
__all__ = ["EdgeModel"]

@ClassFactory.register(ClassType.GENERAL, alias="EdgeModel")
class EdgeModel:
    """Edge-based semantic segmentation model for local deployment.
    
    Implements efficient image segmentation using Segformer architecture.
    Supports pretrained models from Hugging Face and handles all preprocessing steps.

    Attributes:
        model_identity (str): Pretrained model identifier (default: nvidia/segformer-b0-finetuned-cityscapes-1024-1024)
        processor (SegformerImageProcessor): Image preprocessing pipeline
        model (SegformerForSemanticSegmentation): Loaded segmentation model

    Example:
        >>> model = EdgeModel(model_identity="custom/segformer")
        >>> mask = model.predict("test.jpg")
    """
    def __init__(self, **kwargs):
        LOGGER.info("Initializing EdgeModel with kwargs: %s", kwargs)
        self.kwgs = kwargs
        self.model_identity = kwargs.get("model_identity", "nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
        self._set_config()
        self.processor = self._load_processor()
        self.model = None

    def _set_config(self):
        """Set environment variables for model paths."""
        os.environ["model_path"] = self.model_identity

    def _load_processor(self):
        """Load image preprocessing pipeline."""
        return SegformerImageProcessor.from_pretrained(self.model_identity)

    def __call__(self, image_path=None):
        """Factory method for model initialization and prediction.
        
        Args:
            image_path (str, optional): Path to input image. If None, returns initialized model.
        
        Returns:
            Union[EdgeModel, np.ndarray]: Prediction mask.
        """
        if self.model is None:
            self.load()
        if image_path is None:
            return self
        return self.predict(image_path)

    def load(self):
        """Load the segmentation model from specified identity.
        
        Raises:
            RuntimeError: If model loading fails.
        """
        try:
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                self.model_identity,
                ignore_mismatched_sizes=True
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load {self.model_identity}: {str(e)}")

    def predict(self, image_path):
        """Perform semantic segmentation on input image.
        
        Args:
            image_path (str): Path to input image file.
        
        Returns:
            np.ndarray: Predicted segmentation mask.
        """
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        pred_mask = outputs.logits.argmax(dim=1).squeeze().cpu().numpy()
        unique_labels = [CITYSCAPES_CLASSES[i] for i in np.unique(pred_mask) if i < len(CITYSCAPES_CLASSES)]
        LOGGER.info(f"Predicted unique labels: {unique_labels}")
        return pred_mask
=======
version https://git-lfs.github.com/spec/v1
oid sha256:b8387c03555f0aed1bbcc40474af293a48adbcab45459c38ac548d1b5c0fcf39
size 4225
>>>>>>> 9676c3e (ya toh aar ya toh par)
