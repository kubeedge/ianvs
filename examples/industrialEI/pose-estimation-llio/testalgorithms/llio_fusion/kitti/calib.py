import numpy as np


class KittiCalib:
    def __init__(self):
        # Default calibration values for KITTI dataset
        # These are typical values, but should be loaded from calib files
        self.velo2imu = np.array([
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        
        self.imu2velo = np.linalg.inv(self.velo2imu)
        
        # Camera calibration (if needed)
        self.P_rect_02 = None
        self.P_rect_03 = None
        self.R_rect_00 = None
        self.Tr_velo_to_cam = None
        self.Tr_imu_to_velo = None
        
    def load_calib_file(self, calib_file):
        """Load calibration from KITTI calib file"""
        try:
            with open(calib_file, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                key, value = line.split(':', 1)
                key = key.strip()
                value = np.array([float(x) for x in value.split()])
                
                if key == 'P_rect_02':
                    self.P_rect_02 = value.reshape(3, 4)
                elif key == 'P_rect_03':
                    self.P_rect_03 = value.reshape(3, 4)
                elif key == 'R_rect_00':
                    self.R_rect_00 = value.reshape(3, 3)
                elif key == 'Tr_velo_to_cam':
                    self.Tr_velo_to_cam = value.reshape(3, 4)
                elif key == 'Tr_imu_to_velo':
                    self.Tr_imu_to_velo = value.reshape(3, 4)
                    
        except Exception as e:
            print(f"Warning: Could not load calibration file {calib_file}: {e}")
            print("Using default calibration values") 