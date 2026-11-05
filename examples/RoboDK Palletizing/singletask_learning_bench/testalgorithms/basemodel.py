# Copyright 2022 The KubeEdge Authors.
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
import tempfile
import zipfile
import logging
import yaml
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
from ultralytics.utils import LOGGER
from sedna.common.class_factory import ClassType, ClassFactory
from sedna.common.config import Context
from sedna.common.file_ops import FileOps

__all__ = ["BaseModel"]

os.environ['BACKEND_TYPE'] = 'TORCH'


@ClassFactory.register(ClassType.GENERAL, alias="YOLOv8n")
class BaseModel:
    """YOLOv8 Model for Ianvs Framework"""
    def __init__(self, **kwargs):
        """
        Initialize the YOLOv8n algorithm.
        
        Args:
            **kwargs: Hyperparameters from algorithm configuration
        """
        # Model configuration
        self.learning_rate = kwargs.get("learning_rate", 0.01)
        self.epochs = kwargs.get("epochs", 100)
        self.batch_size = kwargs.get("batch_size", 16)
        self.image_size = int(kwargs.get("infer_size", 640))
        self.conf_threshold = float(kwargs.get("conf_threshold", 0.25)) # 0.25

        self.device = kwargs.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_classes = 1
        self.class_names = ["part"]

        self.yolo_run_dir = Path(os.getcwd()) / "runs" / "detect"
        self.best_weight_path = None
        self.workspace = Path(kwargs.get("workspace", "./workspace"))
        self.workspace.mkdir(parents=True, exist_ok=True)
        
        # Paths
        self.checkpoint_path = None
        self.temp_yaml_path = None
        
        # Initialize model
        model_name = "yolov8n.pt"
        self.model = YOLO(model_name)
        
        # Load pretrained model if specified
        initial_model_url = Context.get_parameters("initial_model_url")
        if initial_model_url:
            self.load(initial_model_url)
        
        logging.info(f"YOLOv8n initialized with {self.num_classes} classes")
    
    
    def train(self, train_data, valid_data=None, **kwargs):
        """
        Train the YOLOv8 model without using data.yaml file
        Args:
            train_data: 
                train dataset from testenv configuration
            valid_data: 
                test dataset from testenv configuration
        
        returns:
            self.checkpoint_path: 
                the optimal weight parameter file for the model
        
        """
        if train_data is None or train_data.x is None or train_data.y is None:
            raise Exception("Train data is None.")
        
        # Update configuration from kwargs
        self.epochs = kwargs.get("epochs", self.epochs)
        self.batch_size = kwargs.get("batch_size", self.batch_size)
        self.learning_rate = kwargs.get("learning_rate", self.learning_rate)
        
        logging.info(f"Starting training for {self.epochs} epochs")
        
        # np.array to list
        if isinstance(train_data.x, np.ndarray):
            train_data.x = train_data.x.tolist()
        if isinstance(train_data.y, np.ndarray):
            train_data.y = train_data.y.tolist()
        if valid_data is not None:
            if isinstance(valid_data.x, np.ndarray):
                valid_data.x = valid_data.x.tolist()
            if isinstance(valid_data.y, np.ndarray):
                valid_data.y = valid_data.y.tolist()

        try:
            # Method: Direct data dictionary (no yaml file needed!)
            if hasattr(train_data, 'x') and hasattr(train_data, 'y'):
                # Prepare data in the format YOLOv8 expects
                data_yaml_path = self._prepare_data_dict(train_data, valid_data)
                
                # Train directly with data dictionary
                self.model.train(
                    data=data_yaml_path,  # Direct dictionary instead of yaml path
                    epochs=self.epochs,
                    batch=self.batch_size,
                    lr0=self.learning_rate,
                    imgsz=self.image_size,
                    device=self.device,
                    save=True,
                    project=str(self.workspace),
                    name="yolov8_train",
                    val=True,
                    verbose=True,
                    conf=self.conf_threshold
                )
            else:
                raise Exception("Train data must have 'x' (images) and 'y' (labels) attributes")
            
            train_dirs = list(self.workspace.glob("yolov8_train*"))
            if not train_dirs:
                raise FileNotFoundError(f"No training directories found in {self.workspace.absolute()}")
    
            latest_train_dir = max(train_dirs, key=lambda d: os.path.getctime(str(d)))
            logging.info(f"[Train]Latest training directory: {latest_train_dir.absolute()}")

            self.best_weight_path = latest_train_dir / "weights" / "best.pt"
            if not self.best_weight_path.exists():
                self.best_weight_path = latest_train_dir / "weights" / "last.pt"
                if not self.best_weight_path.exists():
                    raise FileNotFoundError(f"Training failed: No weights found")
                logging.info(f"Warning: Using 'last.pt' instead of 'best.pt'")

        
            logging.info(f"[Train]Training completed! Loaded weight: {self.best_weight_path.absolute()}")
            
            self.checkpoint_path = str(self.best_weight_path)
            logging.info(f"Training completed. Best model saved at: {self.checkpoint_path}")
            return self.checkpoint_path
            
        except Exception as e:
            logging.error(f"Training failed: {str(e)}")
            raise
    
    def _prepare_data_dict(self, train_data, valid_data=None):
        """
        Prepare data dictionary for YOLOv8 training without yaml file
        Args:
            train_data: 
                train dataset from testenv configuration
            valid_data: 
                test dataset from testenv configuration, default None
        
        returns:
            str(data_yaml_path): 
                data.yaml file that meets the requirements of YOLOv8 algorithm
        
        """
        
        # Extract image and label paths
        train_images = train_data.x if hasattr(train_data, 'x') else []
        train_labels = train_data.y if hasattr(train_data, 'y') else []
        
        # Prepare validation data if available
        val_images = valid_data.x if hasattr(valid_data, 'x') else []
        val_labels = valid_data.y if hasattr(valid_data, 'y') else []
        
        # quick check: ensure strings
        train_images = [str(p) for p in train_images]
        val_images = [str(p) for p in val_images]

        # create temp dir and write train.txt / val.txt / data.yaml
        tmp_dir = Path(tempfile.mkdtemp(prefix="yolo_data_"))
        train_txt = tmp_dir / "train.txt"
        val_txt = tmp_dir / "val.txt"

        with open(train_txt, "w") as f:
            f.write("\n".join(train_images))

        val_list = val_images if val_images else train_images
        with open(val_txt, "w") as f:
            f.write("\n".join(val_list))

        data_yaml = {
            "train": str(train_txt),
            "val": str(val_txt),
            "nc": len(self.class_names),
            "names": self.class_names,
        }
        data_yaml_path = tmp_dir / "data.yaml"
        with open(data_yaml_path, "w") as f:
            yaml.safe_dump(data_yaml, f)

        logging.info(f"Temporary data.yaml saved: {data_yaml_path}")
        return str(data_yaml_path)
    

    def save(self, save_path=None, **kwargs):
        """
        Save the yolov8_best.pt file
        Args:
            save_path: 
                the yolov8_best.pt file save path
            **kwargs: 
                this place is of no use
        
        returns:
            dst: 
                the path of the yolov8_best.pt file

        """
        if self.best_weight_path is None or not self.best_weight_path.exists():
            raise RuntimeError("No trained weight found!")

        try:
            if save_path is None:
                save_dir = self.workspace / "weights"
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / "yolov8_best.pt"
            else:
                save_path = Path(save_path)
                if save_path.suffix == "":
                    save_dir = save_path
                    save_dir.mkdir(parents=True, exist_ok=True)
                    save_path = save_dir / "yolov8_best.pt"
                else:
                    save_dir = save_path.parent
                    save_dir.mkdir(parents=True, exist_ok=True)

            src = str(self.best_weight_path.absolute())
            dst = str(save_path.absolute())
            shutil.copyfile(src, dst)
            logging.info(f"Model saved successfully to: {dst}")
            return dst
        except Exception as e:
            raise RuntimeError(
                f"Failed to save model weight:\n"
                f"Source: {self.best_weight_path.absolute()}\n"
                f"Destination: {save_path.absolute()}\n"
                f"Error: {str(e)}"
            ) from e
    
    def predict(self, data, **kwargs):
        """
        Make predictions on the data
        
        Args:
            data: 
                test data from the testenv.yaml, the system has already allocated 

            **kwargs: 
                Hyperparameters from __init__ configuration
        
        returns:
            predictions: dict
                The prediction list corresponding to each image
                eg: {'image_path': pred_list}

        """
        if data is None or (isinstance(data, (list, tuple, dict, str)) and len(data) == 0):
            return []
        
        if isinstance(data, np.ndarray):
            if data.size == 0:
                return []
            # if numpy, convert to list first
            data = data.tolist()

        # Handle different input formats
        if isinstance(data, (list, tuple)):
            # List of image paths
            image_paths = data
        
        elif hasattr(data, '__iter__') and not isinstance(data, (str, dict)):
            # Iterable of data samples
            image_paths = []
            for item in data:
                if isinstance(item, (list, tuple)) and len(item) > 0:
                    # Take first element if it's a tuple/list
                    image_paths.append(item[0])
                elif isinstance(item, str):
                    image_paths.append(item)
                elif isinstance(item, dict) and 'image_path' in item:
                    image_paths.append(item['image_path'])
                else:
                    image_paths.append(str(item))
        else:
            # Single image or unsupported format
            image_paths = [data] if isinstance(data, str) else [str(data)]
        
        # Configure prediction parameters
        conf_threshold = kwargs.get('conf', 0.25)
        iou_threshold = kwargs.get('iou', 0.45)
        max_det = kwargs.get('max_det', 300)
        
        # Make predictions
        try:
            results = self.model.predict(
                source=image_paths,
                conf=conf_threshold,
                iou=iou_threshold,
                max_det=max_det,
                device=self.device,
                save=False,
                **kwargs
            )
            
            # Process results
            predictions = {}
            for result, img_path in zip(results, image_paths):
                img_abs_path = str(Path(img_path).absolute())
                # safe check for boxes
                boxes = []
                scores = []
                classes = []
                if getattr(result, 'boxes', None) is not None and len(result.boxes) > 0:
                    # result.boxes.xyxy might be tensor
                    try:
                        boxes = result.boxes.xyxy.cpu().numpy().tolist()
                        scores = result.boxes.conf.cpu().numpy().tolist()
                        classes = result.boxes.cls.cpu().numpy().astype(int).tolist()
                    except Exception:
                        # fallback: construct empty lists
                        boxes, scores, classes = [], [], []

                # combine the predictions
                pred_list = []
                for cls, score, box in zip(classes, scores, boxes):
                    x1, y1, x2, y2 = box
                    pred_list.append([x1, y1, x2, y2, score, cls])

                predictions[img_abs_path] = pred_list
            
            return predictions
            
        except Exception as e:
            logging.error(f"Prediction failed: {str(e)}")
            raise
    
    def load(self, model_path=None, **kwargs):
        """
        load the save model, if not, load the yolov8n.pt online

        Args:
            model_path: existing yolovx.pt file path

            **kwargs: 
                just to comply with the model specifications, it has no practical use
        
        returns:
            self.model
                saved yolovx.pt 
        """
        if model_path is None:
            logging.info("Loading pre-trained weight: yolov8n.pt")
            self.model = YOLO("yolov8n.pt")
            return
        
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model weight not found: {model_path.absolute()}")
        if model_path.suffix not in [".pt", ".onnx", ".engine"]:
            raise TypeError(f"Unsupported weight format: {model_path.suffix}")
        
        self.model = YOLO(str(model_path))
        logging.info(f"Successfully loaded weight: {model_path.absolute()}")
        return self.model

    
    def evaluate(self, data, model_path, **kwargs):
        """
        Evaluate the model on test data following ianvs framework standard

        Args:
            data: test data from the testenv.yaml, the system has already allocated
            model_path: existing yolovx.pt file path

            **kwargs: 
                to get the metric func and metric name
        
        returns:
            {metric_name: metric_result}: dict
                map50: value
        
        """
        if data is None or data.x is None or data.y is None:
            raise Exception("Prediction data is None")
        
        # Check if the lengths of x and y are consistent
        if len(data.x) != len(data.y):
            raise ValueError(f"Length mismatch: data.x has {len(data.x)} elements, data.y has {len(data.y)} elements")

        # Load the model
        self.load(model_path)
        
        # np.array to list
        if isinstance(data.x, np.ndarray):
            data.x = data.x.tolist()
        if isinstance(data.y, np.ndarray):
            data.y = data.y.tolist()

        # Make predictions
        # 1. predct
        predict_dict = self.predict(data.x)
        
        # Get metric function from kwargs
        metric = kwargs.get("metric")
        if not metric:
            raise ValueError("Metric not provided in kwargs for evaluation.")
        metric_name, metric_func = metric
        
        if callable(metric_func):
            # Call the metric function with ground truth and predictions
            metric_result = metric_func(data.y, predict_dict)
            
            # Return result in the expected format
            # You can customize the metric name based on your needs
            return {metric_name: metric_result}
        else:
            raise Exception(f"not found model metric func(name={metric_name}) in model eval phase")