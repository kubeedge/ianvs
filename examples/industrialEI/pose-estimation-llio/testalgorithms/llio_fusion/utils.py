import os
import datetime
import numpy as np
import copy
import yaml
from functools import wraps
import time

MATPLOTLIB_AVAILABLE = False
OPEN3D_AVAILABLE = False


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(
            f'Function {func.__name__} consumed {total_time:.4f} seconds')
        return result
    return timeit_wrapper


def load_yaml(fpath):
    with open(fpath) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    return cfg


def downsample_points(vel, cfg):
    if OPEN3D_AVAILABLE:
        return velo2downpcd(vel, cfg["voxel_size"])
    else:
        print("Warning: Open3D not available, returning original point cloud")
        return vel


def visualize(vis_material):
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: Matplotlib not available, visualization disabled")
        return
        
    cfg = vis_material["cfg"]
    cfg_path = vis_material["cfg_path"]
    poses_gt = vis_material["poses_gt"]
    poses = vis_material["poses"]
    covs = vis_material["cfg"]

    plt.figure(figsize=(10, 10))
    if is_true(cfg["output"]["plot3d"]):
        print("Visualization mode: 3D")
        ax = plt.axes(projection="3d")
        ax.plot3D(poses[:, 0], poses[:, 1], poses[:, 2], "b")
        ax.plot3D(poses_gt[:, 0], poses_gt[:, 1], poses_gt[:, 2], "r")
    else:
        print("Visualization mode: 2D")
        ax = plt.axes()
        ax.plot(poses[:, 0], poses[:, 1], "b")
        ax.plot(poses_gt[:, 0], poses_gt[:, 1], "r")

        # note: cov is a set of 9x9 matrix in the order of rotation, velocity, and position.
        plot_gaussian(ax, poses[:, 0:2], covs[:, 6:8, 6:8],
                      facecolor=[0.1, 0.3, 1.0], edgecolor=[0, 0, 0],
                      transparency=0.1, sigma=3, upto=1000, skip=10)

    ax.axis('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    if not is_true(cfg["use_lidar_correction"]):
        plt.title("IMU-only Odometry")
        plt.legend(["IMU odometry", "Ground Truth"])
    else:
        plt.title("Open3D Loosely coupled LiDAR+IMU Odometry")
        step = cfg["step_size"]
        plt.legend(
            [f"LiDAR-aided IMU odometry (step: {step})", "Ground Truth"])

    subdirname = time.strftime("%Y-%m-%d-%H-%M-%S")
    savedir = os.path.join(cfg["output"]["save_dir"], subdirname)
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    seq_idx = cfg["input"]["datadrive"]
    figure_save_path = os.path.join(
        savedir, cfg["input"]["dataname"] + f"_{seq_idx}.png")
    plt.savefig(figure_save_path)
    print(f"Saved to {figure_save_path}")

    import shutil
    cfg_src = cfg_path
    cfg_dst = os.path.join(savedir, "cfg.yml")
    shutil.copyfile(cfg_src, cfg_dst)

    # NOTE:
    # at a host-side terminal,
    #  $ xhost +local:docker; is required to visualize the figure
    matplotlib.use("TkAgg")
    plt.show()


def velo2downpcd(velodyne, voxel_size=0.5):
    if not OPEN3D_AVAILABLE:
        print("Warning: Open3D not available, returning original point cloud")
        return velodyne
        
    xyz = velodyne[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return pcd


def velo2skippedpcd(velodyne, skip=20):
    if not OPEN3D_AVAILABLE:
        print("Warning: Open3D not available, returning original point cloud")
        return velodyne
        
    xyz = velodyne[:, :3]
    xyz = xyz[0:xyz.shape[0]:skip, :]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    # pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return pcd


def draw_registration_result(source, target, transformation):
    if not OPEN3D_AVAILABLE:
        print("Warning: Open3D not available, registration visualization disabled")
        return
        
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0, 0])
    target_temp.paint_uniform_color([0, 0, 1])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      "red: scan t-1, blue: scan t")


def is_true(flag):
    if flag == "True" or flag == "true" or flag == "1" or flag == True:
        return True
    else:
        return False


def plot_gaussian(ax, means, covs,
                  edgecolor=[0.0, 0., 1.0], facecolor=[0.0, 0.0, 0.0],
                  transparency=0.5, sigma=3, upto=None, skip=2):
    """Set specific color to show edges, otherwise same with facecolor."""
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: Matplotlib not available, gaussian plotting disabled")
        return

    ellipses = []
    for i in range(len(means)):

        if upto != None and upto < i:
            break

        if i % skip != 0:
            continue

        eigvals, eigvecs = np.linalg.eig(covs[i])

        axis = np.sqrt(eigvals) * sigma
        slope = eigvecs[1][0] / eigvecs[1][1]
        angle = 180.0 * np.arctan(slope) / np.pi
        ellipses.append(Ellipse(means[i, 0:2], axis[0], axis[1], angle=angle))

    facecolor.append(transparency)
    ax.add_collection(PatchCollection(
        ellipses, edgecolors=edgecolor, facecolors=facecolor, linewidth=0.1)) 