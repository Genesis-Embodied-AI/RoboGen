import pybullet as p
import numpy as np
from manipulation.utils import take_round_images_around_object, get_pc
import open3d as o3d
from scipy.spatial.transform import Rotation as R

def voxelize_pc(pc, voxel_size=0.01):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    try:
        voxelized_pcd = pcd.voxel_down_sample(voxel_size)
    except RuntimeError:
        return None
    voxelized_pc = np.asarray(voxelized_pcd.points)
    return voxelized_pc

def rotation_matrix_x(theta):
    """Return a 3x3 rotation matrix for a rotation around the x-axis by angle theta."""
    return R.from_matrix(np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ]))
    
def align_gripper_z_with_normal(normal):
    n_WS = normal
    Gz = n_WS  # gripper z axis aligns with normal # TODO: check the object axis of the franka gripper
    # make orthonormal y axis, aligned with world down
    # y = np.array([0.0, 0.0, -1.0])
    # or, make it horizontal
    y = np.array([0.0, -1, 0])

    Gy = y - np.dot(y, Gz) * Gz
    Gx = np.cross(Gy, Gz)
    R_WG = R.from_matrix(np.vstack((Gx, Gy, Gz)).T)
    return R_WG

def align_gripper_x_with_normal(normal):
    n_WS = normal
    Gx = n_WS  # gripper z axis aligns with normal # TODO: check the object axis of the franka gripper
    # make orthonormal y axis, aligned with world down
    # y = np.array([0.0, 0.0, -1.0])
    # or, make it horizontal
    y = np.array([0.0, -1, 0])
    
    Gy = y - np.dot(y, Gx) * Gx
    Gz = np.cross(Gx, Gy)
    R_WG = R.from_matrix(np.vstack((Gx, Gy, Gz)).T)
    return R_WG


def get_pc_and_normal(simulator, object_name):
    camera_width=640
    camera_height=480
    rgbs, depths, view_camera_matrices, project_camera_matrices = \
        take_round_images_around_object(simulator, object_name, 
                                        return_camera_matrices=True, camera_height=camera_height, camera_width=camera_width, 
                                        only_object=True)
    pcs = []
    for depth, view_matrix, project_matrix in zip(depths, view_camera_matrices, project_camera_matrices):
        pc = get_pc(project_matrix, view_matrix, depth, camera_width, camera_height, mask_infinite=True)
        pcs.append(pc)


    pc = np.concatenate(pcs, axis=0)
    pc = voxelize_pc(pc, voxel_size=0.0005) 

    ### get normals of the point cloud
    pcd = o3d.geometry.PointCloud() 
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd.estimate_normals()
    normals = np.asarray(pcd.normals)

    return pc, normals
