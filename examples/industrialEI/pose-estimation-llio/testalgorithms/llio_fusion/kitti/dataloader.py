from datetime import datetime

import numpy as np

# Minimal stand-in for Dataset using NumPy only
class Data:
    class Dataset:
        def __init__(self):
            pass
        def __len__(self):
            return 0
        def __getitem__(self, i):
            return {}

import pykitti


class KittiDataloader(Data.Dataset):
    def __init__(self, root, dataname, drive, duration=2, step_size=1):
        super().__init__()
        self.duration = duration
        self.data = pykitti.raw(root, dataname, drive)
        self.seq_len = len(self.data.timestamps) - 1

        self.dt = np.array(
            [
                datetime.timestamp(self.data.timestamps[i + 1])
                - datetime.timestamp(self.data.timestamps[i])
                for i in range(self.seq_len)
            ]
        )

        # Loading OXTS data
        
        self.gyro = np.array(
            [
                [
                    self.data.oxts[i].packet.wx,
                    self.data.oxts[i].packet.wy,
                    self.data.oxts[i].packet.wz,
                ]
                for i in range(self.seq_len)
            ]
        )
        self.acc = np.array(
            [
                [
                    self.data.oxts[i].packet.ax,
                    self.data.oxts[i].packet.ay,
                    self.data.oxts[i].packet.az,
                ]
                for i in range(self.seq_len)
            ]
        )
        
        # Simplified rotation handling
        self.gt_rot = np.array(
            [
                [
                    self.data.oxts[i].packet.roll,
                    self.data.oxts[i].packet.pitch,
                    self.data.oxts[i].packet.yaw,
                ]
                for i in range(self.seq_len)
            ]
        )
        # Convert Euler angles to rotation matrices
        from scipy.spatial.transform import Rotation as R
        self.gt_rot_matrices = []
        for euler in self.gt_rot:
            rot_matrix = R.from_euler('xyz', euler).as_matrix()
            self.gt_rot_matrices.append(rot_matrix)
        
        # Simplified velocity calculation
        self.gt_vel = np.array(
            [
                [
                    self.data.oxts[i].packet.vf,
                    self.data.oxts[i].packet.vl,
                    self.data.oxts[i].packet.vu,
                ]
                for i in range(self.seq_len)
            ]
        )
        
        self.gt_pos = np.array([self.data.oxts[i].T_w_imu[0:3, 3]
                               for i in range(self.seq_len)])

        # Loading velodyne data (large memory)
        self.velodyne = [(self.data.get_velo(i)) for i in range(self.seq_len)]
        # Example scan info: len(self.velodyne), self.velodyne[6].shape

        start_frame = 0
        end_frame = self.seq_len

        self.index_map = [
            i for i in range(0, end_frame - start_frame - self.duration, step_size)
        ]
        # print(f"self.index_map is {self.index_map}")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, i):
        frame_id = self.index_map[i]
        end_frame_id = frame_id + self.duration
        
        # Simplified version
        return {
            "dt": self.dt[frame_id:end_frame_id],
            "acc": self.acc[frame_id:end_frame_id],
            "gyro": self.gyro[frame_id:end_frame_id],
            "gt_pos": self.gt_pos[frame_id + 1: end_frame_id + 1],
            "gt_rot": self.gt_rot_matrices[frame_id + 1: end_frame_id + 1],
            "gt_vel": self.gt_vel[frame_id + 1: end_frame_id + 1],
            "velodyne": self.velodyne[frame_id],
            "init_pos": self.gt_pos[frame_id:frame_id+1],
            "init_rot": self.gt_rot_matrices[frame_id:end_frame_id],
            "init_vel": self.gt_vel[frame_id:frame_id+1],
        }

    def get_init_value(self):
        # Simplified version
        return {"pos": self.gt_pos[:1],
                "rot": self.gt_rot_matrices[:1],
                "vel": self.gt_vel[:1],
                "velodyne": self.velodyne[:1]}


def imu_collate(data):
    # NumPy version
    acc = np.stack([d["acc"] for d in data])
    gyro = np.stack([d["gyro"] for d in data])

    gt_pos = np.stack([d["gt_pos"] for d in data])
    gt_rot = np.stack([d["gt_rot"] for d in data])
    gt_vel = np.stack([d["gt_vel"] for d in data])

    init_pos = np.stack([d["init_pos"] for d in data])
    init_rot = np.stack([d["init_rot"] for d in data])
    init_vel = np.stack([d["init_vel"] for d in data])

    dt = np.stack([d["dt"] for d in data]).reshape(-1, 1)

    velodyne = [d["velodyne"] for d in data]

    return {
        "dt": dt,
        "acc": acc,
        "gyro": gyro,
        "gt_pos": gt_pos,
        "gt_vel": gt_vel,
        "gt_rot": gt_rot,
        "velodyne": velodyne,
        "init_pos": init_pos,
        "init_vel": init_vel,
        "init_rot": init_rot,
    } 