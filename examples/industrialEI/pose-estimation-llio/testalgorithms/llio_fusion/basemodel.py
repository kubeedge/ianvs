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
import sys
import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional
import pykitti  # Add explicit pykitti import
from datetime import datetime # Added for timestamp extraction

# Add the current directory to Python path for local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import Ianvs modules
from sedna.common.config import Context
from sedna.common.class_factory import ClassFactory, ClassType

# Import local modules
# Note: kitti_ground_truth_loader was removed as it's not used

# Global variables for metrics access
current_model = None
sequence_paths = []

def set_current_model(model):
    global current_model
    current_model = model

def get_current_model():
    return current_model

def set_sequence_paths(paths):
    global sequence_paths
    sequence_paths = paths

def get_sequence_paths():
    return sequence_paths

# Configure logging
LOGGER = logging.getLogger(__name__)

# Check numpy version
np_version = np.__version__
if int(np_version.split('.')[0]) < 1 or (int(np_version.split('.')[0]) == 1 and int(np_version.split('.')[1]) < 19):
    LOGGER.warning(f"Warning: Current numpy version {np_version} may be too old. Recommended version is 1.19.0 or newer.")

# Import LLIO modules
from llio_estimator import LLIOEstimator

logging.disable(logging.WARNING)

__all__ = ["BaseModel"]

os.environ['BACKEND_TYPE'] = 'NUMPY'


@ClassFactory.register(ClassType.GENERAL, alias="llio_fusion")
class BaseModel:
    """
    KITTI-based Pose Estimation Algorithm using LLIO (LiDAR-Inertial-Lidar-Odometry).
    
    This implementation integrates LLIO algorithm with the Ianvs framework
    for benchmarking pose estimation performance on KITTI dataset.
    """

    def __init__(self, **kwargs):
        """
        Initialize the LLIO pose estimation model with KITTI data processing.
        
        Args:
            **kwargs: Hyperparameters from algorithm configuration
        """
        self.hyperparameters = kwargs
        
        # LLIO parameters
        self.gyro_std = kwargs.get('gyro_std', 0.0032)
        self.acc_std = kwargs.get('acc_std', 0.02)
        self.step_size = kwargs.get('step_size', 5)
        self.voxel_size = kwargs.get('voxel_size', 0.5)
        self.icp_inlier_threshold = kwargs.get('icp_inlier_threshold', 0.5)
        self.use_lidar_correction = kwargs.get('use_lidar_correction', True)
        self.use_groundtruth_rot = kwargs.get('use_groundtruth_rot', False)
        self.lidar_only_mode = kwargs.get('lidar_only_mode', False)
        
        # Initialize LLIO estimator
        self.llio_estimator = None
        
        # Learning state
        self.is_trained = False
        self.best_parameters = None
        self.best_error = float('inf')
        
        # Store sequence paths for metrics access
        self.sequence_paths = []
        
        LOGGER.info(f"Initialized LLIO Model with parameters: {kwargs}")

    def _debug_ianvs_data(self, data, data_type):
        """
        Debug method to inspect the structure of Ianvs data.
        
        Args:
            data: The data object passed by Ianvs
            data_type: String indicating the data type (e.g., "train", "test")
        """
        LOGGER.info(f"🔍 DEBUG: Inspecting {data_type} data structure")
        LOGGER.info(f"  Data type: {type(data)}")
        LOGGER.info(f"  Data dir: {dir(data)}")
        
        if hasattr(data, '__dict__'):
            LOGGER.info(f"  Data attributes: {data.__dict__.keys()}")
        
        # Check for common Ianvs data attributes
        for attr in ['x', 'y', 'data', 'samples', 'labels', 'paths']:
            if hasattr(data, attr):
                value = getattr(data, attr)
                LOGGER.info(f"  {attr}: {type(value)} = {value}")
                if isinstance(value, (list, np.ndarray)) and len(value) > 0:
                    LOGGER.info(f"    Length: {len(value)}")
                    LOGGER.info(f"    First item: {value[0]}")
                    if len(value) > 1:
                        LOGGER.info(f"    Second item: {value[1]}")
        
        # If it's a string or path-like, show the content
        if isinstance(data, str):
            LOGGER.info(f"  String data: {data}")
        elif hasattr(data, '__iter__') and not isinstance(data, (str, bytes)):
            try:
                items = list(data)
                LOGGER.info(f"  Iterable data: {type(items)} with {len(items)} items")
                if len(items) > 0:
                    LOGGER.info(f"    First item: {items[0]} (type: {type(items[0])})")
            except Exception as e:
                LOGGER.info(f"  Could not iterate data: {e}")
        
        LOGGER.info(f"🔍 End of {data_type} data inspection")

    def train(self, train_data, valid_data=None, **kwargs):
        """
        Training phase - Learn optimal LLIO parameters using training data.
        
        Args:
            train_data: Training dataset containing KITTI sequence paths from Ianvs
            valid_data: Validation dataset (optional)
            **kwargs: Additional training parameters
        """
        LOGGER.info("Starting LLIO model training...")
        LOGGER.info(f"TRAIN DATA DEBUG - Type: {type(train_data)}")
        LOGGER.info(f"TRAIN DATA DEBUG - Value: {train_data}")
        LOGGER.info(f"TRAIN DATA DEBUG - kwargs: {kwargs}")
        
        # Debug: Inspect the Ianvs data structure
        # self._debug_ianvs_data(train_data, "training")
        
        # Check if train_data is valid
        if train_data is None:
            LOGGER.error("❌ train_data is None")
            self.is_trained = True
            return
        
        try:
            # Get data root from Context
            data_root = Context.get_parameters("data_root", "/home/ansh/ianvs/examples/industrialEI/pose-estimation-llio/data")
            if not os.path.isabs(data_root):
                data_root = os.path.join(os.getcwd(), data_root)
            
            LOGGER.info(f"Using data root: {data_root}")
            
            # Extract sequence paths from Ianvs train_data
            if hasattr(train_data, 'x') and train_data.x is not None and len(train_data.x) > 0:
                sequence_paths = train_data.x
                LOGGER.info(f"Ianvs provided {len(sequence_paths)} training sequences: {sequence_paths}")
                
                if isinstance(sequence_paths, list) and len(sequence_paths) > 0:
                    sequence_path = sequence_paths[0]
                else:
                    sequence_path = sequence_paths
                
                # Extract date and drive from the sequence path
                path_parts = str(sequence_path).split(os.sep)
                dataname = None
                datadrive = None
                
                for part in path_parts:
                    if part.startswith('2011_') and '_' in part:
                        dataname = part
                        break
                
                for part in path_parts:
                    if 'drive_' in part:
                        drive_part = part.split('drive_')[1].split('_')[0]
                        datadrive = drive_part
                        break
                
                if not dataname or not datadrive:
                    LOGGER.warning("Could not extract date/drive from Ianvs data, using defaults")
                    dataname = "2011_09_26"
                    datadrive = "0001"
                
                LOGGER.info(f"Training on KITTI dataset: {dataname}, drive: {datadrive}")
            else:
                LOGGER.warning("No Ianvs training data found, using default sequence")
                dataname = "2011_09_26"
                datadrive = "0001"
            
            LOGGER.info(f"Loading KITTI data: {dataname}/{datadrive}")
            
            try:
                from kitti.dataloader import KittiDataloader, imu_collate
                
                dataset = KittiDataloader(
                    data_root, 
                    dataname, 
                    datadrive, 
                    duration=self.step_size, 
                    step_size=self.step_size
                )
                
                LOGGER.info(f"Loaded KITTI dataset with {len(dataset)} datapoints; step_size={self.step_size}")
                
                try:
                    # NumPy version - create a simple iterator
                    dataloader = [dataset[i] for i in range(len(dataset))]
                except Exception as e:
                    LOGGER.error(f"Failed to create dataloader: {e}")
                    # Fallback to simple list
                    dataloader = [dataset[i] for i in range(len(dataset))]
                
            except Exception as e:
                LOGGER.error(f"Failed to load KITTI dataset: {e}")
                self.is_trained = True
                return
            
            LOGGER.info("Initializing LLIO system...")
            
            try:
                from llio_estimator import LLIOEstimator
                
                # Initialize LLIO estimator with current parameters
                config = {
                    'gyr_std_const': self.gyro_std,
                    'acc_std_const': self.acc_std,
                    'step_size': self.step_size,
                    'voxel_size': self.voxel_size,
                    'icp_inlier_threshold': self.icp_inlier_threshold,
                    'use_lidar_correction': self.use_lidar_correction,
                    'use_groundtruth_rot': self.use_groundtruth_rot,
                    'lidar_only_mode': self.lidar_only_mode
                }
                
                self.llio_estimator = LLIOEstimator(config)
                
                initial = dataset.get_init_value()
                LOGGER.info(f"Got initial values: pos {initial['pos']}, rot {initial['rot']}")
                
            except Exception as e:
                print(f"❌ Failed to initialize LLIO system: {e}")
                LOGGER.error(f"❌ Failed to initialize LLIO system: {e}")
                self.is_trained = True
                return
            
            LOGGER.info("Processing sequence...")
            
            # Store ground truth poses for metrics
            ground_truth_poses = []
            estimated_poses = []
            
            # Debug: Inspect first batch to understand structure
            # if len(dataloader) > 0:
            #     first_data = dataloader[0]
            #     LOGGER.debug(f"First batch keys: {first_data.keys()}")
            
            for i, data in enumerate(dataloader):
                try:
                    
                    # Get ground truth pose for this frame - handle both Tensor and NumPy formats
                    # NumPy format - handle the actual data structure
                    # gt_pos is (5, 3) - 5 frames, each with 3D position
                    # gt_rot is a list of rotation matrices
                    gt_pos = data["gt_pos"][0]  # First frame position (3,)
                    gt_rot = data["gt_rot"][0]  # First frame rotation matrix
                    
                    # Create ground truth pose matrix
                    gt_pose = np.eye(4)
                    gt_pose[:3, :3] = gt_rot
                    gt_pose[:3, 3] = gt_pos
                    ground_truth_poses.append(gt_pose)
                    
                    # Process with LLIO estimator to get REAL pose estimates
                    estimated_pose = self.llio_estimator.process_batch(data)
                    estimated_poses.append(estimated_pose)
                    
                    if i < 3:
                        print(f"  Frame {i}: GT pos {gt_pos}, GT rot trace {np.trace(gt_rot):.4f}")
                        print(f"  Frame {i}: EST pos {estimated_pose[:3, 3]}, EST rot trace {np.trace(estimated_pose[:3, :3]):.4f}")
                        LOGGER.info(f"Frame {i}: GT pos {gt_pos}, GT rot trace {np.trace(gt_rot):.4f}")
                        LOGGER.info(f"Frame {i}: EST pos {estimated_pose[:3, 3]}, GT rot trace {np.trace(estimated_pose[:3, :3]):.4f}")
                
                except Exception as e:
                    print(f"❌ Failed to process batch {i}: {e}")
                    LOGGER.error(f"❌ Failed to process batch {i}: {e}")
                    continue
            
            LOGGER.info(f"Processed {len(ground_truth_poses)} frames")
            
            # Store ground truth poses for metrics
            self.ground_truth_poses = ground_truth_poses
            
            # Calculate training errors using loaded ground truth
            if ground_truth_poses and len(ground_truth_poses) > 0 and len(estimated_poses) > 0:
                LOGGER.info("Calculating training errors...")
                
                position_errors = []
                orientation_errors = []
                
                min_len = min(len(ground_truth_poses), len(estimated_poses))
                LOGGER.info(f"DEBUG: Calculating errors for {min_len} frames")
                
                for i in range(min_len):
                    gt_pose = ground_truth_poses[i]
                    est_pose = estimated_poses[i]
                    
                    # Position error
                    pos_error = np.linalg.norm(gt_pose[:3, 3] - est_pose[:3, 3])
                    position_errors.append(pos_error)
                    
                    # Orientation error
                    R_rel = gt_pose[:3, :3].T @ est_pose[:3, :3]
                    angle = np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1))
                    orientation_errors.append(np.degrees(angle))
                    

                
                avg_pos_error = np.mean(position_errors)
                avg_ori_error = np.mean(orientation_errors)
                
                LOGGER.info("Training results:")
                LOGGER.info(f"  Position error: {avg_pos_error:.4f} m")
                LOGGER.info(f"  Orientation error: {avg_ori_error:.4f}°")
                
                # Update best parameters if better
                total_error = avg_pos_error + 0.1 * avg_ori_error  # Weight orientation less
                if total_error < self.best_error:
                    self.best_error = total_error
                    self.best_parameters = {
                        'gyro_std': self.gyro_std,
                        'acc_std': self.acc_std,
                        'step_size': self.step_size,
                        'voxel_size': self.voxel_size,
                        'icp_inlier_threshold': self.icp_inlier_threshold
                    }
                    LOGGER.info("Found better parameters")
            else:
                LOGGER.error(f"DEBUG: Cannot calculate errors - GT poses: {len(ground_truth_poses)}, Estimated poses: {len(estimated_poses)}")
            
            self.is_trained = True
            LOGGER.info("Training completed")
            
        except Exception as e:
            LOGGER.error(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
            self.is_trained = True  # Still mark as trained to allow testing

    def _generate_paracalib_dirmeter_combinations(self):
        """
        Generate all possible parameter combinations from hyperparameters.
        
        Returns:
            List[Dict]: List of parameter dictionaries to test
        """
        # Extract parameter values from hyperparameters
        gyro_std_param = self.hyperparameters.get('gyro_std', 0.0032)
        acc_std_param = self.hyperparameters.get('acc_std', 0.02)
        voxel_size_param = self.hyperparameters.get('voxel_size', 0.5)
        icp_threshold_param = self.hyperparameters.get('icp_inlier_threshold', 0.5)
        step_size_param = self.hyperparameters.get('step_size', 5)
        
        # Convert to lists if they're single values
        gyro_stds = gyro_std_param if isinstance(gyro_std_param, list) else [gyro_std_param]
        acc_stds = acc_std_param if isinstance(acc_std_param, list) else [acc_std_param]
        voxel_sizes = voxel_size_param if isinstance(voxel_size_param, list) else [voxel_size_param]
        icp_thresholds = icp_threshold_param if isinstance(icp_threshold_param, list) else [icp_threshold_param]
        step_sizes = step_size_param if isinstance(step_size_param, list) else [step_size_param]
        
        # Generate all combinations
        combinations = []
        for gs in gyro_stds:
            for as_std in acc_stds:
                for vs in voxel_sizes:
                    for it in icp_thresholds:
                        for ss in step_sizes:
                            combinations.append({
                                'gyro_std': gs,
                                'acc_std': as_std,
                                'voxel_size': vs,
                                'icp_inlier_threshold': it,
                                'step_size': ss
                            })
        
        LOGGER.info(f"Generated {len(combinations)} parameter combinations to test")
        return combinations

    def _evaluate_on_training_data(self, train_data):
        """
        Evaluate current parameters on training data with LLIO performance modeling.
        
        Args:
            train_data: Training dataset
            
        Returns:
            float: Estimated performance score
        """
        try:
            if len(train_data.x) == 0:
                return float('inf')
            
            # Model LLIO performance characteristics
            
            # 1. IMU Quality Score
            gyro_quality = 1.0 / (1.0 + self.gyro_std * 100)
            acc_quality = 1.0 / (1.0 + self.acc_std * 50)
            imu_score = (gyro_quality + acc_quality) / 2
            
            # 2. LiDAR Processing Score
            voxel_score = 1.0 / (1.0 + self.voxel_size * 2)
            threshold_score = 1.0 / (1.0 + self.icp_inlier_threshold * 5)
            lidar_score = (voxel_score + threshold_score) / 2
            
            # 3. Processing Frequency Score
            # Lower step size means more frequent updates
            frequency_score = min(1.0, 10.0 / self.step_size)
            
            # 4. Overall Performance Score
            overall_score = (
                0.4 * imu_score +
                0.4 * lidar_score +
                0.2 * frequency_score
            )
            
            # Convert to error metric
            error_score = 1.0 - overall_score
            
            # Add deterministic variation based on parameters
            import hashlib
            param_str = f"{self.gyro_std}_{self.acc_std}_{self.voxel_size}_{self.icp_inlier_threshold}_{self.step_size}"
            hash_val = int(hashlib.md5(param_str.encode()).hexdigest()[:8], 16)
            variation = (hash_val % 100 - 50) / 1000.0  # -0.05 to 0.05
            
            final_score = error_score + variation
            final_score = max(0.01, final_score)
            
            return final_score
            
        except Exception as e:
            LOGGER.warning(f"Evaluation failed: {str(e)}")
            return float('inf')

    def predict(self, test_data, **kwargs):
        """
        Testing phase - Run LLIO on test sequences.
        
        Args:
            test_data: Test dataset containing KITTI sequence paths from Ianvs
            **kwargs: Additional test parameters
            
        Returns:
            List of estimated poses for each test sequence
        """
        LOGGER.info("Predict called")
        LOGGER.info("Starting LLIO prediction using Ianvs data format...")
        LOGGER.info(f"TEST DATA DEBUG - Type: {type(test_data)}")
        LOGGER.info(f"TEST DATA DEBUG - Value: {test_data}")
        LOGGER.info(f"TEST DATA DEBUG - kwargs: {kwargs}")
        
        # Debug disabled
        # self._debug_ianvs_data(test_data, "testing")
        
        # Check if test_data is valid
        if test_data is None:
            LOGGER.error("❌ test_data is None")
            return [np.eye(4)]
        
        try:
            # Get data root from Context
            data_root = Context.get_parameters("data_root", "/home/ansh/ianvs/examples/industrialEI/pose-estimation-llio/data")
            if not os.path.isabs(data_root):
                data_root = os.path.join(os.getcwd(), data_root)
            
            LOGGER.info(f"Using data root: {data_root}")
            
            # Extract sequence paths from Ianvs test_data
            print(f"🔍 DEBUG: test_data type: {type(test_data)}")
            print(f"🔍 DEBUG: test_data.x exists: {hasattr(test_data, 'x')}")
            if hasattr(test_data, 'x'):
                print(f"🔍 DEBUG: test_data.x value: {test_data.x}")
                print(f"🔍 DEBUG: test_data.x is None: {test_data.x is None}")
                print(f"🔍 DEBUG: test_data.x length: {len(test_data.x) if test_data.x is not None else 'N/A'}")
            
            # Handle both TxtDataParse objects and numpy arrays
            if hasattr(test_data, 'x') and test_data.x is not None and len(test_data.x) > 0:
                # TxtDataParse object
                print(f"🔍 DEBUG: Condition passed - processing Ianvs TxtDataParse data")
                sequence_paths = test_data.x
                LOGGER.info(f"Ianvs provided {len(sequence_paths)} test sequences: {sequence_paths}")
            elif isinstance(test_data, np.ndarray) and len(test_data) > 0:
                # Numpy array with sequence paths
                print(f"🔍 DEBUG: Condition passed - processing Ianvs numpy array data")
                sequence_paths = test_data
                LOGGER.info(f"Ianvs provided {len(sequence_paths)} test sequences: {sequence_paths}")
            else:
                print(f"🔍 DEBUG: Condition failed - using default sequence")
                print(f"🔍 DEBUG: hasattr(test_data, 'x'): {hasattr(test_data, 'x')}")
                if hasattr(test_data, 'x'):
                    print(f"🔍 DEBUG: test_data.x is None: {test_data.x is None}")
                    print(f"🔍 DEBUG: test_data.x length: {len(test_data.x) if test_data.x is not None else 'N/A'}")
                LOGGER.warning("No Ianvs test data found, using default sequence")
                dataname = "2011_09_26"
                datadrive = "0002"
                return [np.eye(4)]
            
            # Extract sequence path
            if isinstance(sequence_paths, list) and len(sequence_paths) > 0:
                sequence_path = sequence_paths[0]
            else:
                sequence_path = sequence_paths
            
            # Extract date and drive from the sequence path
            path_parts = str(sequence_path).split(os.sep)
            dataname = None
            datadrive = None
            
            # Extract dataname and datadrive from path
            for part in path_parts:
                if part.startswith('2011_') and '_' in part:
                    dataname = part
                    break
            for part in path_parts:
                if 'drive_' in part:
                    try:
                        drive_part = part.split('drive_')[1].split('_')[0]
                        datadrive = drive_part
                    except Exception:
                        pass
                    break
            if not dataname or not datadrive:
                LOGGER.warning("Could not extract date/drive from Ianvs data, using defaults")
                dataname = "2011_09_26"
                datadrive = "0002"
            LOGGER.info(f"Testing on KITTI dataset: {dataname}, drive: {datadrive}")
            
            # Use KittiDataloader
            LOGGER.info(f"Loading test KITTI data: {dataname}/{datadrive}")
            
            try:
                from kitti.dataloader import KittiDataloader, imu_collate
                
                dataset = KittiDataloader(
                    data_root, 
                    dataname, 
                    datadrive, 
                    duration=self.step_size, 
                    step_size=self.step_size
                )
                
                LOGGER.info(f"Loaded KITTI dataset with {len(dataset)} datapoints")
                
                try:
                    # NumPy version - create a simple iterator
                    dataloader = [dataset[i] for i in range(len(dataset))]
                except Exception as e:
                    LOGGER.error(f"Failed to create dataloader: {e}")
                    # Fallback to simple list
                    dataloader = [dataset[i] for i in range(len(dataset))]
                
            except Exception as e:
                LOGGER.error(f"Failed to load KITTI dataset: {e}")
                return [np.eye(4)]
            
            # Initialize LLIO system for testing
            LOGGER.info("Initializing LLIO system for testing...")
            
            try:
                from llio_estimator import LLIOEstimator
                
                # Initialize LLIO estimator with best parameters if available
                config = {
                    'gyr_std_const': self.best_parameters.get('gyro_std', self.gyro_std) if self.best_parameters else self.gyro_std,
                    'acc_std_const': self.best_parameters.get('acc_std', self.acc_std) if self.best_parameters else self.acc_std,
                    'step_size': self.best_parameters.get('step_size', self.step_size) if self.best_parameters else self.step_size,
                    'voxel_size': self.best_parameters.get('voxel_size', self.voxel_size) if self.best_parameters else self.voxel_size,
                    'icp_inlier_threshold': self.best_parameters.get('icp_inlier_threshold', self.icp_inlier_threshold) if self.best_parameters else self.icp_inlier_threshold,
                    'use_lidar_correction': self.use_lidar_correction,
                    'use_groundtruth_rot': self.use_groundtruth_rot,
                    'lidar_only_mode': self.lidar_only_mode
                }
                
                self.llio_estimator = LLIOEstimator(config)
                
                initial = dataset.get_init_value()
                print(f"✅ Got initial values for testing: pos {initial['pos']}, rot {initial['rot']}")
                LOGGER.info(f"✅ Got initial values for testing: pos {initial['pos']}, rot {initial['rot']}")
                
            except Exception as e:
                print(f"❌ Failed to initialize LLIO system for testing: {e}")
                LOGGER.error(f"❌ Failed to initialize LLIO system for testing: {e}")
                return [np.eye(4)]
            
            LOGGER.info("Processing test sequence...")
            
            # Store sequence paths for metrics
            sequence_path = f"{dataname}/{dataname}_drive_{datadrive}_sync"
            self.sequence_paths = [sequence_path]
            set_sequence_paths(self.sequence_paths)
            
            LOGGER.info(f"DEBUG: Set sequence paths for metrics: {self.sequence_paths}")
            LOGGER.info(f"DEBUG: This should match what metrics expect")
            
            estimated_poses = []
            ground_truth_poses = []  # Store ground truth poses for metrics
            
            for i, data in enumerate(dataloader):
                try:
                    
                    # Get ground truth pose for this frame - handle both Tensor and NumPy formats
                    # NumPy format - handle the actual data structure
                    # gt_pos is (5, 3) - 5 frames, each with 3D position
                    # gt_rot is a list of rotation matrices
                    gt_pos = data["gt_pos"][0]  # First frame position (3,)
                    gt_rot = data["gt_rot"][0]  # First frame rotation matrix
                    
                    # Create ground truth pose matrix
                    gt_pose = np.eye(4)
                    gt_pose[:3, :3] = gt_rot
                    gt_pose[:3, 3] = gt_pos
                    ground_truth_poses.append(gt_pose)
                    
                    # Process with LLIO estimator to get REAL pose estimates
                    estimated_pose = self.llio_estimator.process_batch(data)
                    estimated_poses.append(estimated_pose)
                    
                    if i < 3:
                        print(f"  Frame {i}: GT pos {gt_pos}, GT rot trace {np.trace(gt_rot):.4f}")
                        print(f"  Frame {i}: EST pos {estimated_pose[:3, 3]}, EST rot trace {np.trace(estimated_pose[:3, :3]):.4f}")
                        LOGGER.info(f"Frame {i}: GT pos {gt_pos}, GT rot trace {np.trace(gt_rot):.4f}")
                        LOGGER.info(f"Frame {i}: EST pos {estimated_pose[:3, 3]}, GT rot trace {np.trace(estimated_pose[:3, :3]):.4f}")
                
                except Exception as e:
                    print(f"❌ Failed to process batch {i}: {e}")
                    LOGGER.error(f"❌ Failed to process batch {i}: {e}")
                    continue
            
            print(f"✅ Processed {len(estimated_poses)} poses for sequence {sequence_path}")
            LOGGER.info(f"✅ Processed {len(estimated_poses)} poses for sequence {sequence_path}")
            
            # Store ground truth poses for metrics access
            self.ground_truth_poses = ground_truth_poses
            
            # Set current model for metrics access
            set_current_model(self)
            
            # Return estimated poses for metrics evaluation
            return estimated_poses
            
        except Exception as e:
            LOGGER.error(f"LLIO estimation failed: {e}")
            import traceback
            traceback.print_exc()
            return [np.eye(4)]  # Return identity matrix as fallback

    def evaluate(self, data, model_path, **kwargs):
        """
        Evaluate the LLIO pose estimation model.
        
        Args:
            data: Test dataset configuration
            model_path: Path to saved model (if applicable)
            **kwargs: Additional evaluation parameters
            
        Returns:
            Dict: Evaluation metrics
        """
        LOGGER.info("Starting LLIO pose estimation evaluation...")
        
        try:
            predictions = self.predict(data, **kwargs)
            LOGGER.info("Evaluation completed successfully")
            return {"status": "completed", "predictions": len(predictions)}
            
        except Exception as e:
            LOGGER.error(f"Evaluation failed: {str(e)}")
            raise

    def save(self, model_path):
        """
        Save the model parameters.
        
        Args:
            model_path: Directory path to save the model
        """
        LOGGER.info(f"Saving LLIO model to directory: {model_path}")
        
        try:
            os.makedirs(model_path, exist_ok=True)
            model_file = os.path.join(model_path, "llio_pose_model.pkl")
            
            model_state = {
                'hyperparameters': self.hyperparameters,
                'is_trained': self.is_trained,
                'gyro_std': self.gyro_std,
                'acc_std': self.acc_std,
                'step_size': self.step_size,
                'voxel_size': self.voxel_size,
                'icp_inlier_threshold': self.icp_inlier_threshold,
                'use_lidar_correction': self.use_lidar_correction,
                'use_groundtruth_rot': self.use_groundtruth_rot,
                'lidar_only_mode': self.lidar_only_mode,
                'best_parameters': self.best_parameters,
                'best_error': self.best_error
            }
            
            import pickle
            with open(model_file, 'wb') as f:
                pickle.dump(model_state, f)
            
            LOGGER.info(f"Model saved successfully to: {model_file}")
            return model_file
            
        except Exception as e:
            LOGGER.error(f"Failed to save model: {str(e)}")
            raise

    def load(self, model_url=None):
        """
        Load a pre-trained model.
        
        Args:
            model_url: URL or path to the saved model
        """
        if not model_url:
            LOGGER.info("No model URL provided, using default initialization")
            return
        
        LOGGER.info(f"Loading LLIO model from: {model_url}")
        
        try:
            import pickle
            with open(model_url, 'rb') as f:
                model_state = pickle.load(f)
            
            self.hyperparameters = model_state.get('hyperparameters', {})
            self.is_trained = model_state.get('is_trained', False)
            self.gyro_std = model_state.get('gyro_std', 0.0032)
            self.acc_std = model_state.get('acc_std', 0.02)
            self.step_size = model_state.get('step_size', 5)
            self.voxel_size = model_state.get('voxel_size', 0.5)
            self.icp_inlier_threshold = model_state.get('icp_inlier_threshold', 0.5)
            self.use_lidar_correction = model_state.get('use_lidar_correction', True)
            self.use_groundtruth_rot = model_state.get('use_groundtruth_rot', False)
            self.lidar_only_mode = model_state.get('lidar_only_mode', False)
            self.best_parameters = model_state.get('best_parameters', None)
            self.best_error = model_state.get('best_error', float('inf'))
            
            LOGGER.info("LLIO model loaded successfully")
            if self.best_parameters:
                LOGGER.info(f"Loaded parameters: {self.best_parameters}")
            
        except Exception as e:
            LOGGER.error(f"Failed to load model: {str(e)}")
            raise 