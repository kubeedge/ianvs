<<<<<<< HEAD
"""
LLIO Fusion Algorithm Package for Ianvs Benchmarking.

This package provides LiDAR-Inertial-Lidar-Odometry (LLIO) implementation
for pose estimation benchmarking on KITTI dataset.
"""

__version__ = "1.0.0"
__author__ = "LLIO Team"

# Core algorithm imports
from .basemodel import BaseModel
from .llio_estimator import LLIOEstimator

# Utility imports
from .utils import *
from .kitti.dataloader import KittiDataloader, imu_collate
from .kitti.calib import KittiCalib

__all__ = [
    "BaseModel",
    "LLIOEstimator", 
    "KittiDataloader",
    "imu_collate",
    "KittiCalib"
] 
=======
version https://git-lfs.github.com/spec/v1
oid sha256:f8a6a5e713a3298abbf541854e5632e8bde1b6b1b334685a632de3c4c5903a52
size 585
>>>>>>> 9676c3e (ya toh aar ya toh par)
