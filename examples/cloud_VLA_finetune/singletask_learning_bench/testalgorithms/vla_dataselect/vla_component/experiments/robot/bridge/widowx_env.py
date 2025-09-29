"""
WidowXGym environment definition.

Copied over from the Octo eval code linked below, with some modifications:
https://github.com/octo-models/octo/blob/main/examples/envs/widowx_env.py
"""

import time
from typing import Dict

import gym
import numpy as np
from pyquaternion import Quaternion
from widowx_envs.widowx_env_service import WidowXClient


def state_to_eep(xyz_coor, zangle: float):
    """
    Implements the state to end-effector pose function, returning a 4x4 matrix.
    Refer to `widowx_controller/widowx_controller.py` in the `bridge_data_robot` codebase.
    """
    assert len(xyz_coor) == 3
    DEFAULT_ROTATION = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])
    new_pose = np.eye(4)
    new_pose[:3, -1] = xyz_coor
    new_quat = Quaternion(axis=np.array([0.0, 0.0, 1.0]), angle=zangle) * Quaternion(matrix=DEFAULT_ROTATION)
    new_pose[:3, :3] = new_quat.rotation_matrix
    return new_pose


def wait_for_obs(widowx_client):
    """Fetches an observation from the WidowXClient."""
    obs = widowx_client.get_observation()
    while obs is None:
        print("Waiting for observations...")
        obs = widowx_client.get_observation()
        time.sleep(1)
    return obs


def convert_obs(obs, im_size):
    """Preprocesses image and proprio observations."""
    # Preprocess image
    image_obs = (obs["image"].reshape(3, im_size, im_size).transpose(1, 2, 0) * 255).astype(np.uint8)
    # Add padding to proprio to match RLDS training
    proprio = np.concatenate([obs["state"][:6], [0], obs["state"][-1:]])
    return {
        "image_primary": image_obs,
        "full_image": obs["full_image"],
        "proprio": proprio,
    }


def null_obs(img_size):
    """Returns a dummy observation with all-zero image and proprio."""
    return {
        "image_primary": np.zeros((img_size, img_size, 3), dtype=np.uint8),
        "proprio": np.zeros((8,), dtype=np.float64),
    }


class WidowXGym(gym.Env):
    """
    A Gym environment for the WidowX controller provided by:
    https://github.com/rail-berkeley/bridge_data_robot
    """

    def __init__(
        self,
        widowx_client: WidowXClient,
        cfg: Dict,
        im_size: int = 256,
        blocking: bool = True,
    ):
        self.widowx_client = widowx_client
        self.im_size = im_size
        self.blocking = blocking
        self.observation_space = gym.spaces.Dict(
            {
                "image_primary": gym.spaces.Box(
                    low=np.zeros((im_size, im_size, 3)),
                    high=255 * np.ones((im_size, im_size, 3)),
                    dtype=np.uint8,
                ),
                "full_image": gym.spaces.Box(
                    low=np.zeros((480, 640, 3)),
                    high=255 * np.ones((480, 640, 3)),
                    dtype=np.uint8,
                ),
                "proprio": gym.spaces.Box(low=np.ones((8,)) * -1, high=np.ones((8,)), dtype=np.float64),
            }
        )
        self.action_space = gym.spaces.Box(low=np.zeros((7,)), high=np.ones((7,)), dtype=np.float64)
        self.cfg = cfg

    def step(self, action):
        self.widowx_client.step_action(action, blocking=self.blocking)

        raw_obs = self.widowx_client.get_observation()

        truncated = False
        if raw_obs is None:
            # this indicates a loss of connection with the server
            # due to an exception in the last step so end the trajectory
            truncated = True
            obs = null_obs(self.im_size)  # obs with all zeros
        else:
            obs = convert_obs(raw_obs, self.im_size)

        return obs, 0, False, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.widowx_client.reset()
        self.move_to_start_state()

        raw_obs = wait_for_obs(self.widowx_client)
        obs = convert_obs(raw_obs, self.im_size)

        return obs, {}

    def get_observation(self):
        raw_obs = wait_for_obs(self.widowx_client)
        obs = convert_obs(raw_obs, self.im_size)
        return obs

    def move_to_start_state(self):
        successful = False
        while not successful:
            try:
                # Get XYZ position from user.
                init_x, init_y, init_z = self.cfg.init_ee_pos
                x_val = input(f"Enter x value of gripper starting position (leave empty for default == {init_x}): ")
                if x_val == "":
                    x_val = init_x
                y_val = input(f"Enter y value of gripper starting position (leave empty for default == {init_y}): ")
                if y_val == "":
                    y_val = init_y
                z_val = input(f"Enter z value of gripper starting position (leave empty for default == {init_z}): ")
                if z_val == "":
                    z_val = init_z
                # Fix initial orientation and add user's commanded XYZ into start transform.
                # Initial orientation: gripper points ~15 degrees away from the standard orientation (quat=[0, 0, 0, 1]).
                transform = np.array(
                    [
                        [0.267, 0.000, 0.963, float(x_val)],
                        [0.000, 1.000, 0.000, float(y_val)],
                        [-0.963, 0.000, 0.267, float(z_val)],
                        [0.00, 0.00, 0.00, 1.00],
                    ]
                )
                # IMPORTANT: It is very important to move to reset position with blocking==True.
                #            Otherwise, the controller's `_reset_previous_qpos()` call will be called immediately after
                #            the move command is given -- and before the move is complete -- and the initial state will
                #            be totally incorrect.
                self.widowx_client.move(transform, duration=0.8, blocking=True)
                successful = True
            except Exception as e:
                print(e)
