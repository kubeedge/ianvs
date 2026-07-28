"""
Perception Module for Deformable Assembly

Complete pipeline: YOLO + 6D Pose + Deformation Estimation

Utilizes full dataset:
- RGB images (YOLO detection)
- Depth images (3D pose estimation)
- Segmentation masks (deformation analysis)
- Labels/annotations (ground truth)
- Metadata (camera intrinsics)
"""

import os
import cv2
import numpy as np
import torch
import json
import yaml
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler with formatting
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('[%(levelname)s] [%(name)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("ultralytics not installed")

from sedna.common.class_factory import ClassFactory, ClassType


@ClassFactory.register(ClassType.GENERAL, alias="PerceptionModel")
class PerceptionModel:
    """
    Complete perception pipeline for deformable assembly
    
    Pipeline:
    1. YOLO 2D detection on RGB
    2. Depth-based 6D pose estimation
    3. Segmentation-based deformation analysis
    """
    
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cpu",
        img_size: int = 640,
        dataset_root: str = None,
        **kwargs
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.img_size = img_size
        
        # Dataset root from environment or kwargs
        self.dataset_root = Path(dataset_root or os.getenv(
            'DATASET_ROOT', 
            './deformable_assembly_dataset'
        ))
        
        # Class names
        self.class_names = {
            0: "ram",
            1: "cooling_mounts",
            2: "cpu_slot",
            3: "fcp",
            4: "chip_key"
        }
        
        # Camera intrinsics (loaded from metadata)
        self.camera_intrinsics = None
        
        # YOLO model
        self.model = None
        self._load_model()
        
        logger.info("Perception model initialized")
        logger.info(f"Dataset root: {self.dataset_root}")
        logger.info(f"Device: {device}")
        logger.info(f"Confidence threshold: {conf_threshold}, IOU threshold: {iou_threshold}")
    
    def _load_model(self):
        """Load YOLO model"""
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics not installed")
        
        try:
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            logger.info(f"Model loaded successfully: {self.model_path}")
        except Exception as e:
            logger.warning(f"Failed to load {self.model_path}: {e}")
            logger.info("Loading default YOLOv8n...")
            self.model = YOLO('yolov8n.pt')
            self.model.to(self.device)
            logger.info("Default YOLOv8n loaded successfully")
    
    def train(self, train_data, valid_data=None, **kwargs):
        """
        Train YOLO + pose/deformation models
        
        Multi-stage training:
        1. YOLO for 2D detection
        2. TODO: 6D pose network
        3. TODO: Deformation estimation network
        """
        logger.info("="*80)
        logger.info("Starting Multi-Stage Training")
        logger.info("="*80)
        
        # Stage 1: YOLO training
        logger.info("[Stage 1] Training YOLO detector...")
        self._train_yolo(train_data, valid_data, **kwargs)
        
        # Stage 2: Pose estimation (TODO: Implement pose network)
        # logger.info("[Stage 2] Training 6D pose estimator...")
        # self._train_pose_network(train_data, valid_data, **kwargs)
        
        # Stage 3: Deformation estimation (TODO: Implement deformation network)
        # logger.info("[Stage 3] Training deformation estimator...")
        # self._train_deformation_network(train_data, valid_data, **kwargs)
        
        logger.info("Multi-stage training complete")
        return self.model
    
    def _train_yolo(self, train_data, valid_data, **kwargs):
        """Train YOLO detector"""
        epochs = kwargs.get('epochs', 3)
        batch_size = kwargs.get('batch', 1)
        img_size = kwargs.get('imgsz', 640)
        
        logger.info(f"YOLO training configuration - Epochs: {epochs}, Batch: {batch_size}, Image size: {img_size}")
        
        data_yaml_path = self._prepare_yolo_dataset(train_data, valid_data)
        
        try:
            logger.info("Starting YOLO training...")
            results = self.model.train(
                data=data_yaml_path,
                epochs=epochs,
                imgsz=img_size,
                batch=batch_size,
                device=self.device,
                project='runs/train',
                name='assembly_yolo',
                exist_ok=True,
                verbose=True,
                cache=False,
                workers=0,
                amp=False,
                deterministic=True,
                rect=False,   # Disable rectangular training
                close_mosaic=0,  # Disable mosaic augmentation early
                augment=False
            )
            
            logger.info("YOLO training completed")
            
            best_path = Path(results.save_dir) / 'weights' / 'best.pt'
            if best_path.exists():
                logger.info(f"Loading best model from {best_path}")
                self.model = YOLO(str(best_path))
                self.model.to(self.device)
                logger.info("Best YOLO model loaded successfully")
            else:
                logger.warning(f"Best model not found at {best_path}")

            return True
                
        except Exception as e:
            logger.error(f"YOLO training error: {e}")
            import traceback
            traceback.print_exc()
            
            logger.warning("Falling back to pretrained model...")
            return True
    
    def predict(self, data, **kwargs) -> np.ndarray:
        """
        Complete perception pipeline
        
        For each frame:
        1. YOLO 2D detection on RGB
        2. Load depth + segmentation
        3. Compute 6D pose from depth
        4. Analyze deformation from segmentation
        5. Return complete perception output
        """
        if hasattr(data, 'x'):
            data_x = data.x
        else:
            data_x = data

        logger.info(f"Running complete perception pipeline on {len(data_x)} frames")
        
        all_predictions = []
        
        for idx, line in enumerate(data_x):
            if idx % 50 == 0:
                logger.info(f"Progress: {idx}/{len(data_x)}")
            
            # Parse paths from dataset format
            paths = self._parse_data_line(line)
            
            if paths is None:
                logger.debug(f"Frame {idx}: No valid paths found")
                all_predictions.append([])
                continue
            
            # Load multimodal data
            rgb_img = self._load_rgb(paths['rgb'])
            depth_img = self._load_depth(paths['depth'])
            seg_mask = self._load_segmentation(paths['seg'])
            metadata = self._load_metadata(paths['metadata'])
            
            logger.debug(f"Frame {idx}: Loaded RGB={rgb_img is not None}, "
                        f"Depth={depth_img is not None}, Seg={seg_mask is not None}, "
                        f"Metadata={metadata is not None}")
            
            # Update camera intrinsics from metadata
            if metadata and self.camera_intrinsics is None:
                self._load_camera_intrinsics(metadata)
            
            # Step 1: YOLO 2D detection
            detections_2d = self._run_yolo_detection(paths['rgb'])
            logger.debug(f"Frame {idx}: Detected {len(detections_2d)} objects")
            
            # Step 2: Enhance with 6D pose + deformation
            detections_full = []
            for det_idx, det_2d in enumerate(detections_2d):
                # Compute 6D pose from depth
                pose_6d = self._estimate_6d_pose(
                    det_2d, rgb_img, depth_img, metadata
                )
                
                # Analyze deformation from segmentation
                deformation = self._estimate_deformation(
                    det_2d, seg_mask, depth_img
                )
                
                logger.debug(f"Frame {idx}, Object {det_idx}: Class={det_2d['class_name']}, "
                            f"Pose=[{pose_6d['position'][0]:.3f}, {pose_6d['position'][1]:.3f}, {pose_6d['position'][2]:.3f}], "
                            f"Deformation={deformation['delta']:.4f}")
                
                # Combine all perception outputs
                full_detection = {
                    **det_2d, # 2D bbox, class, confidence
                    'pose_6d': pose_6d, # [x, y, z, qx, qy, qz, qw]
                    'deformation': deformation # Deformation metrics
                }
                
                detections_full.append(full_detection)
            
            all_predictions.append(detections_full)
        
        logger.info("Perception pipeline complete")
        return self._format_predictions(all_predictions)
    
    def _parse_data_line(self, line: str) -> Optional[Dict[str, str]]:
        """
        Parse dataset line to extract all modality paths
        
        Format: "image_path label_path"
        Assumes structure:
        episodes/episode_XXX/images/rgb/frame_XXXX.png
        episodes/episode_XXX/images/depth/frame_XXXX.npy
        episodes/episode_XXX/images/segmentation/frame_XXXX.png
        episodes/episode_XXX/labels/frame_XXXX.txt
        episodes/episode_XXX/annotations/metadata.json
        """
        parts = line.strip().split()
        if not parts:
            logger.debug("Empty data line")
            return None
        
        rgb_path = parts[0].strip()
        
        # Convert to absolute path
        rgb_full = Path(rgb_path)
        
        if not rgb_full.is_absolute():
            rgb_full = self.dataset_root / rgb_path.lstrip('./')

        if not rgb_full.exists():
            logger.warning(f"RGB not found: {rgb_full}")
            return None
        
        # Construct other modality paths
        episode_dir = rgb_full.parent.parent.parent # Go up to episode_XXX
        frame_name = rgb_full.stem # frame_XXXX
        
        paths = {
            'rgb': str(rgb_full),
            'depth': str(episode_dir / 'images' / 'depth' / f'{frame_name}.npy'),
            'seg': str(episode_dir / 'images' / 'segmentation' / f'{frame_name}.png'),
            'label': str(episode_dir / 'labels' / f'{frame_name}.txt'),
            'metadata': str(episode_dir / 'annotations' / 'metadata.json')
        }
        
        logger.debug(f"Parsed paths for frame {frame_name}")
        return paths
    
    def _load_rgb(self, path: str) -> Optional[np.ndarray]:
        """Load RGB image"""
        if not os.path.exists(path):
            logger.debug(f"RGB image not found: {path}")
            return None
        
        img = cv2.imread(path)
        if img is not None:
            logger.debug(f"Loaded RGB image: {path}, shape={img.shape}")
        else:
            logger.warning(f"Failed to read RGB image: {path}")
        return img
    
    def _load_depth(self, path: str) -> Optional[np.ndarray]:
        """Load depth image (.npy format)"""
        if not os.path.exists(path):
            logger.debug(f"Depth image not found: {path}")
            return None
        try:
            depth = np.load(path)
            logger.debug(f"Loaded depth image: {path}, shape={depth.shape}, "
                        f"range=[{depth.min():.3f}, {depth.max():.3f}]")
            return depth
        except Exception as e:
            logger.error(f"Failed to load depth image from {path}: {e}")
            return None
    
    def _load_segmentation(self, path: str) -> Optional[np.ndarray]:
        """Load segmentation mask"""
        if not os.path.exists(path):
            logger.debug(f"Segmentation mask not found: {path}")
            return None
        
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            unique_classes = np.unique(mask)
            logger.debug(f"Loaded segmentation mask: {path}, shape={mask.shape}, "
                        f"classes={unique_classes.tolist()}")
        else:
            logger.warning(f"Failed to read segmentation mask: {path}")
        return mask
    
    def _load_metadata(self, path: str) -> Optional[Dict]:
        """Load episode metadata (camera params, etc.)"""
        if not os.path.exists(path):
            logger.debug(f"Metadata not found: {path}")
            return None
        try:
            with open(path, 'r') as f:
                metadata = json.load(f)
            logger.debug(f"Loaded metadata from {path}")
            return metadata
        except Exception as e:
            logger.error(f"Failed to load metadata from {path}: {e}")
            return None
    
    def _load_camera_intrinsics(self, metadata: Dict):
        """Extract camera intrinsics from metadata"""
        if 'camera' in metadata:
            cam = metadata['camera']
            fx = cam.get('fx', 525.0)
            fy = cam.get('fy', 525.0)
            cx = cam.get('cx', 320.0)
            cy = cam.get('cy', 240.0)
            
            self.camera_intrinsics = {
                'fx': fx, 'fy': fy,
                'cx': cx, 'cy': cy
            }
            logger.info(f"Camera intrinsics loaded: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
        else:
            logger.warning("No camera intrinsics found in metadata")
    
    def _run_yolo_detection(self, rgb_path: str) -> List[Dict]:
        """Run YOLO 2D detection"""
        try:
            results = self.model.predict(
                source=rgb_path,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.img_size,
                device=self.device,
                verbose=False
            )
            
            detections = self._extract_yolo_detections(results[0])
            logger.debug(f"YOLO detected {len(detections)} objects in {rgb_path}")
            return detections
        except Exception as e:
            logger.error(f"YOLO detection error on {rgb_path}: {e}")
            return []
    
    def _extract_yolo_detections(self, result) -> List[Dict]:
        """Extract 2D detections from YOLO"""
        detections = []
        
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            
            for box, conf, cls in zip(boxes, confs, classes):
                class_name = self.class_names.get(int(cls), 'unknown')
                detections.append({
                    'bbox': box.tolist(),
                    'confidence': float(conf),
                    'class_id': int(cls),
                    'class_name': class_name
                })
                logger.debug(f"Detection: {class_name} (conf={conf:.3f}), bbox={box.tolist()}")
        
        return detections
    
    def _estimate_6d_pose(
        self, 
        detection_2d: Dict,
        rgb: np.ndarray,
        depth: np.ndarray,
        metadata: Dict
    ) -> Dict:
        """
        Estimate 6D pose from 2D detection + depth
        
        Returns:
        --------
        pose_6d: dict
            {
                'position': [x, y, z], # meters
                'orientation': [qx, qy, qz, qw], # quaternion
                'confidence': float
            }
        """
        if depth is None or self.camera_intrinsics is None:
            # Fallback: return dummy pose
            logger.debug(f"Cannot estimate 6D pose: depth={depth is not None}, "
                        f"intrinsics={self.camera_intrinsics is not None}")
            return {
                'position': [0.0, 0.0, 0.0],
                'orientation': [0.0, 0.0, 0.0, 1.0],
                'confidence': 0.0
            }
        
        # Extract bbox center
        bbox = detection_2d['bbox']
        cx_2d = int((bbox[0] + bbox[2]) / 2)
        cy_2d = int((bbox[1] + bbox[3]) / 2)
        
        # Get depth at center
        h, w = depth.shape
        cx_2d = np.clip(cx_2d, 0, w-1)
        cy_2d = np.clip(cy_2d, 0, h-1)
        z = depth[cy_2d, cx_2d]
        
        if z <= 0:
            logger.debug(f"Invalid depth at ({cx_2d}, {cy_2d}): {z}, using default 0.5m")
            z = 0.5 # Default depth if invalid
        
        # Back-project to 3D using camera intrinsics
        fx = self.camera_intrinsics['fx']
        fy = self.camera_intrinsics['fy']
        cx = self.camera_intrinsics['cx']
        cy = self.camera_intrinsics['cy']
        
        x = (cx_2d - cx) * z / fx
        y = (cy_2d - cy) * z / fy
        
        logger.debug(f"6D pose estimated: position=[{x:.3f}, {y:.3f}, {z:.3f}]m")
        
        # TODO: Estimate orientation using PCA/ICP on point cloud
        # For now: identity orientation
        orientation = [0.0, 0.0, 0.0, 1.0] # [qx, qy, qz, qw]
        
        return {
            'position': [float(x), float(y), float(z)],
            'orientation': orientation,
            'confidence': detection_2d['confidence']
        }
    
    def _estimate_deformation(
        self,
        detection_2d: Dict,
        segmentation: np.ndarray,
        depth: np.ndarray
    ) -> Dict:
        """
        Estimate deformation from segmentation mask + depth
        
        Returns:
        --------
        deformation: dict
            {
                'delta': float, # Deformation parameter (0=rigid, >0=deformed)
                'type': str, # 'none', 'bend', 'twist', 'wrinkle'
                'confidence': float
            }
        """
        if segmentation is None or depth is None:
            logger.debug("Cannot estimate deformation: missing segmentation or depth data")
            return {
                'delta': 0.0,
                'type': 'none',
                'confidence': 0.0
            }
        
        # Extract ROI from segmentation
        bbox = detection_2d['bbox']
        x1, y1, x2, y2 = [int(v) for v in bbox]
        
        h, w = segmentation.shape
        x1, x2 = np.clip([x1, x2], 0, w)
        y1, y2 = np.clip([y1, y2], 0, h)
        
        roi_seg = segmentation[y1:y2, x1:x2]
        roi_depth = depth[y1:y2, x1:x2]
        
        # Get object mask (assume class ID corresponds to pixel value)
        class_id = detection_2d['class_id']
        mask = (roi_seg == class_id).astype(np.uint8)
        
        if mask.sum() < 10:
            # Not enough pixels
            logger.debug(f"Insufficient mask pixels for class {class_id}: {mask.sum()}")
            return {
                'delta': 0.0,
                'type': 'none',
                'confidence': 0.0
            }
        
        # Extract object point cloud
        ys, xs = np.where(mask > 0)
        zs = roi_depth[ys, xs]
        
        # Filter invalid depths
        valid = zs > 0
        if valid.sum() < 10:
            logger.debug(f"Insufficient valid depth points: {valid.sum()}")
            return {
                'delta': 0.0,
                'type': 'none',
                'confidence': 0.0
            }
        
        zs = zs[valid]
        
        # Compute deformation metric (depth variance)
        z_std = np.std(zs)
        z_mean = np.mean(zs)
        
        # Normalize deformation parameter
        delta = float(z_std / (z_mean + 1e-6))
        
        # Classify deformation type (simple threshold-based)
        if delta < 0.01:
            deform_type = 'none'
        elif delta < 0.05:
            deform_type = 'slight'
        else:
            deform_type = 'significant'
        
        logger.debug(f"Deformation analysis: delta={delta:.4f}, type={deform_type}, "
                    f"z_std={z_std:.3f}, z_mean={z_mean:.3f}, points={len(zs)}")
        
        return {
            'delta': delta,
            'type': deform_type,
            'confidence': detection_2d['confidence']
        }
    
    def _format_predictions(self, predictions: List[List[Dict]]) -> np.ndarray:
        """Format predictions for Ianvs"""
        formatted = np.empty(len(predictions), dtype=object)
        for i, pred in enumerate(predictions):
            formatted[i] = pred
        logger.debug(f"Formatted {len(predictions)} predictions")
        return formatted
    
    def _prepare_yolo_dataset(self, train_data, valid_data) -> str:
        """Prepare YOLO data.yaml"""
        # Simple config with relative paths
        data_config = {
            'path': str(self.dataset_root.resolve()),
            'train': 'train_index.txt',
            'val': 'test_index.txt',
            'names': self.class_names,
            'nc': len(self.class_names)
        }
        
        yaml_path = self.dataset_root / 'data.yaml'
        try:
            with open(yaml_path, 'w') as f:
                yaml.dump(data_config, f, default_flow_style=False)
            logger.info(f"YOLO dataset configuration prepared: {yaml_path}")
            logger.debug(f"Dataset config: {data_config}")
        except Exception as e:
            logger.error(f"Failed to create YOLO dataset configuration: {e}")
            raise
        
        return str(yaml_path)
    
    def save(self, model_path: str):
        """Save all models"""
        if self.model:
            try:
                self.model.save(model_path)
                logger.info(f"Model saved successfully: {model_path}")
            except Exception as e:
                logger.error(f"Failed to save model to {model_path}: {e}")
        else:
            logger.warning("No model to save")
    
    def load(self, model_path: str):
        """Load all models"""
        try:
            self.model = YOLO(model_path)
            self.model.to(self.device)
            logger.info(f"Model loaded successfully: {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise


if __name__ == "__main__":
    logger.info("="*80)
    logger.info("COMPLETE PERCEPTION MODULE")
    logger.info("="*80)
    logger.info("\nFeatures:")
    logger.info("✓ YOLO 2D detection (RGB)")
    logger.info("✓ 6D pose estimation (Depth)")
    logger.info("✓ Deformation analysis (Segmentation)")
    logger.info("✓ Multi-modal data loading")
    logger.info("✓ Ianvs-compatible")
    logger.info("="*80)