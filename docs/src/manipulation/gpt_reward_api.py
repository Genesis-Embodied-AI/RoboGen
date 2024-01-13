import pybullet as p
import numpy as np
from manipulation.utils import get_pc
from manipulation.grasping_utils import get_pc_and_normal
from manipulation.utils import take_round_images
from gpt_4.query import query
import os
from scipy import ndimage

def compute_obj_to_center_dist(simulator, obj_a, obj_b):
    obj_a_center = get_position(simulator, obj_a)
    obj_b_bbox_min, obj_b_bbox_max = get_bounding_box(simulator, obj_b)
    obj_b_center = (obj_b_bbox_min + obj_b_bbox_max) / 2
    return np.linalg.norm(obj_a_center - obj_b_center)

def get_initial_joint_angle(simulator, object_name, joint_name):
    object_name = object_name.lower()
    return simulator.initial_joint_angle[object_name][joint_name]

def get_initial_pos_orient(simulator, object_name):
    object_name = object_name.lower()
    return simulator.initial_pos[object_name], np.array(p.getEulerFromQuaternion(simulator.initial_orient[object_name]))

### success check functions
def gripper_close_to_object_link(simulator, object_name, link_name):
    link_pc = get_link_pc(simulator, object_name, link_name)
    gripper_pos, _ = get_eef_pos(simulator)
    distance = np.linalg.norm(link_pc.reshape(-1, 3) - gripper_pos.reshape(1, 3), axis=1)
    if np.min(distance) < 0.06:
        return True
    return False

def gripper_close_to_object(simulator, object_name):
    object_pc, _ = get_pc_and_normal(simulator, object_name)
    gripper_pos, _ = get_eef_pos(simulator)
    distance = np.linalg.norm(object_pc.reshape(-1, 3) - gripper_pos.reshape(1, 3), axis=1)
    if np.min(distance) < 0.06:
        return True
    return False

def check_grasped(self, object_name, link_name=None):
    object_name = object_name.lower()
    grasped_object_name, grasped_link_name = get_grasped_object_and_link_name(self)
    if link_name is None:
        return grasped_object_name == object_name
    else:
        return grasped_object_name == object_name and grasped_link_name == link_name

def get_grasped_object_name(simulator):
    grasped_object_id = simulator.suction_obj_id
    if grasped_object_id is None:
        return None
    
    id_to_name = {v: k for k, v in simulator.urdf_ids.items()}
    return id_to_name[grasped_object_id]

def get_grasped_object_and_link_name(simulator):
    grasped_object_id = simulator.suction_obj_id
    grasped_link_id = simulator.suction_contact_link
    if grasped_object_id is None or grasped_link_id is None:
        return None, None
    
    id_to_name = {v: k for k, v in simulator.urdf_ids.items()}
    grasped_obj_name = id_to_name[grasped_object_id]

    if grasped_link_id == -1:
        return grasped_obj_name, "base"
    
    joint_info = p.getJointInfo(grasped_object_id, grasped_link_id, physicsClientId=simulator.id)
    link_name = joint_info[12].decode("utf-8")

    return grasped_obj_name, link_name

def get_joint_limit(simulator, object_name, custom_joint_name):
    object_name = object_name.lower()
    object_id = simulator.urdf_ids[object_name]
    num_joints = p.getNumJoints(object_id, physicsClientId=simulator.id)
    
    urdf_joint_name = custom_joint_name
    max_joint_val = 0
    min_joint_val = 0
    for j_id in range(num_joints):
        joint_info = p.getJointInfo(object_id, j_id, physicsClientId=simulator.id)
        if joint_info[1].decode("utf-8") == urdf_joint_name:
            max_joint_val = joint_info[9]
            min_joint_val = joint_info[8]
            break
    
    if min_joint_val < max_joint_val:
        return min_joint_val, max_joint_val
    else:
        return max_joint_val, min_joint_val

def get_position(simulator, object_name):
    object_name = object_name.lower()
    object_id = simulator.urdf_ids[object_name]
    return np.array(p.getBasePositionAndOrientation(object_id, physicsClientId=simulator.id)[0])

def get_velocity(simulator, object_name):
    object_name = object_name.lower()
    object_id = simulator.urdf_ids[object_name]
    return np.array(p.getBaseVelocity(object_id, physicsClientId=simulator.id)[0])

def get_orientation(simulator, object_name):
    object_name = object_name.lower()
    object_id = simulator.urdf_ids[object_name]
    return np.array(p.getEulerFromQuaternion(p.getBasePositionAndOrientation(object_id, physicsClientId=simulator.id)[1]))

def get_eef_pos(simulator):
    robot_eef_pos, robot_eef_orient = simulator.robot.get_pos_orient(simulator.robot.right_end_effector)
    return np.array(robot_eef_pos).flatten(), np.array(p.getEulerFromQuaternion(robot_eef_orient)).flatten()

def get_finger_pos(simulator):
    left_finger_joint_pos =  p.getLinkState(simulator.robot.body, simulator.robot.right_gripper_indices[0], physicsClientId=simulator.id)[0]
    right_finger_joint_pos = p.getLinkState(simulator.robot.body, simulator.robot.right_gripper_indices[1], physicsClientId=simulator.id)[0]
    return np.array(left_finger_joint_pos), np.array(right_finger_joint_pos)

def get_finger_distance(simulator): 
    left_finger_joint_angle = p.getJointState(simulator.robot.body, simulator.robot.right_gripper_indices[0], physicsClientId=simulator.id)[0]
    right_finger_joint_angle = p.getJointState(simulator.robot.body, simulator.robot.right_gripper_indices[1], physicsClientId=simulator.id)[0]
    return left_finger_joint_angle + right_finger_joint_angle

def get_bounding_box(simulator, object_name):
    object_name = object_name.lower()
    object_id = simulator.urdf_ids[object_name]
    if object_name != "init_table":
        return simulator.get_aabb(object_id)
    else:
        return simulator.table_bbox_min, simulator.table_bbox_max

def get_bounding_box_link(simulator, object_name, link_name):
    object_name = object_name.lower()
    link_id = get_link_id_from_name(simulator, object_name, link_name)
    object_id = simulator.urdf_ids[object_name]
    return simulator.get_aabb_link(object_id, link_id)
    
def get_joint_state(simulator, object_name, custom_joint_name):
    object_name = object_name.lower()
    object_id = simulator.urdf_ids[object_name]
    num_joints = p.getNumJoints(object_id, physicsClientId=simulator.id)

    urdf_joint_name = custom_joint_name
    for i in range(num_joints):
        joint_info = p.getJointInfo(object_id, i, physicsClientId=simulator.id)
        if joint_info[1].decode("utf-8") == urdf_joint_name:
            joint_index = i
            break
        
    return np.array(p.getJointState(object_id, joint_index, physicsClientId=simulator.id)[0])

def get_link_state(simulator, object_name, custom_link_name):
    object_name = object_name.lower()
    object_id = simulator.urdf_ids[object_name]
    urdf_link_name = custom_link_name
    link_id = get_link_id_from_name(simulator, object_name, urdf_link_name)
    link_pos, link_orient = p.getLinkState(object_id, link_id, physicsClientId=simulator.id)[:2]
    return np.array(link_pos)
    

def get_link_pc(simulator, object_name, custom_link_name):
    object_name = object_name.lower()
    urdf_link_name = custom_link_name 
    link_com, all_pc = render_to_get_link_com(simulator, object_name, urdf_link_name)

    return all_pc

def set_joint_value(simulator, object_name, joint_name, joint_value="max"):
    object_name = object_name.lower()
    object_id = simulator.urdf_ids[object_name]
    num_joints = p.getNumJoints(object_id, physicsClientId=simulator.id)
    joint_index = None
    max_joint_val = 0
    min_joint_val = 0
    for j_id in range(num_joints):
        joint_info = p.getJointInfo(object_id, j_id, physicsClientId=simulator.id)
        print(joint_info[1])
        if joint_info[1].decode("utf-8") == joint_name:
            joint_index = j_id
            max_joint_val = joint_info[9]
            min_joint_val = joint_info[8]
            break
    
    if joint_value == 'max':
        p.resetJointState(object_id, joint_index, max_joint_val, physicsClientId=simulator.id)
    elif joint_value == 'min':
        p.resetJointState(object_id, joint_index, min_joint_val, physicsClientId=simulator.id)
    else:
        p.resetJointState(object_id, joint_index, joint_value, physicsClientId=simulator.id)


def in_bbox(simulator, pos, bbox_min, bbox_max):
    if (pos[0] <= bbox_max[0] and pos[0] >= bbox_min[0] and \
        pos[1] <= bbox_max[1] and pos[1] >= bbox_min[1] and \
        pos[2] <= bbox_max[2] and pos[2] >= bbox_min[2]):
        return True
    return False

def grasped(simulator, object_name):
    if object_name in simulator.grasped_object_list:
        return True
    return False

def render_to_get_link_com(simulator, object_name, urdf_link_name):    
    ### make all other objects invisiable
    prev_rgbas = []
    object_id = simulator.urdf_ids[object_name]
    for obj_name, obj_id in simulator.urdf_ids.items():
        if obj_name != object_name:
            num_links = p.getNumJoints(obj_id, physicsClientId=simulator.id)
            for link_idx in range(-1, num_links):
                prev_rgba = p.getVisualShapeData(obj_id, link_idx, physicsClientId=simulator.id)[0][14:18]
                prev_rgbas.append(prev_rgba)
                p.changeVisualShape(obj_id, link_idx, rgbaColor=[0, 0, 0, 0], physicsClientId=simulator.id)

    ### center camera to the target object
    env_prev_view_matrix, env_prev_projection_matrix = simulator.view_matrix, simulator.projection_matrix
    camera_width = 640
    camera_height = 480
    obj_id = object_id
    min_aabb, max_aabb = simulator.get_aabb(obj_id)
    camera_target = (max_aabb + min_aabb) / 2
    distance = np.linalg.norm(max_aabb - min_aabb) * 1.2
    elevation = 30

    ### get a round of images of the target object
    rgbs, depths, view_matrices, projection_matrices = take_round_images(
        simulator, camera_target, distance, elevation, 
        camera_width=camera_width, camera_height=camera_height, 
        z_near=0.01, z_far=10,
        return_camera_matrices=True)

    ### make the target link invisiable
    link_id = get_link_id_from_name(simulator, object_name, urdf_link_name)
    # import pdb; pdb.set_trace()
    prev_link_rgba = p.getVisualShapeData(obj_id, link_id, physicsClientId=simulator.id)[0][14:18]
    p.changeVisualShape(obj_id, link_id, rgbaColor=[0, 0, 0, 0], physicsClientId=simulator.id)

    ### get a round of images of the target object with link invisiable
    rgbs_link_invisiable, _ = take_round_images(
        simulator, camera_target, distance, elevation,
        camera_width=camera_width, camera_height=camera_height, 
        z_near=0.01, z_far=10,
    )

    ### use subtraction to get the link mask
    max_num_diff_pixels = 0
    best_idx = 0
    for idx, (rgb, rgb_link_invisiable) in enumerate(zip(rgbs, rgbs_link_invisiable)):
        diff_image = np.abs(rgb - rgb_link_invisiable)
        diff_pixels = np.sum(diff_image > 0)
        if diff_pixels > max_num_diff_pixels:
            max_num_diff_pixels = diff_pixels
            best_idx = idx

    best_mask = np.abs(rgbs[best_idx] - rgbs_link_invisiable[best_idx]) > 0
    best_mask = np.any(best_mask, axis=2)


    ### get the link mask center
    center = ndimage.measurements.center_of_mass(best_mask)
    center = [int(center[0]), int(center[1])]

    ### back project the link mask center to get the link com in 3d coordinate
    best_pc = get_pc(projection_matrices[best_idx], view_matrices[best_idx], depths[best_idx], camera_width, camera_height)
    
    pt_idx = center[0] * camera_width + center[1]
    link_com = best_pc[pt_idx]
    best_pc = best_pc.reshape((camera_height, camera_width, 3))
    all_pc = best_pc[best_mask]


    ### reset the object and link rgba to previous values, and the simulator view matrix and projection matrix
    p.changeVisualShape(obj_id, link_id, rgbaColor=prev_link_rgba, physicsClientId=simulator.id)

    cnt = 0
    object_id = simulator.urdf_ids[object_name]
    for obj_name, obj_id in simulator.urdf_ids.items():
        if obj_name != object_name:
            num_links = p.getNumJoints(obj_id, physicsClientId=simulator.id)
            for link_idx in range(-1, num_links):
                p.changeVisualShape(obj_id, link_idx, rgbaColor=prev_rgbas[cnt], physicsClientId=simulator.id)
                cnt += 1

    simulator.view_matrix, simulator.projection_matrix = env_prev_view_matrix, env_prev_projection_matrix

    ### add a safety check here in case the rendering fails
    bounding_box = get_bounding_box_link(simulator, object_name, urdf_link_name)
    if not in_bbox(simulator, link_com, bounding_box[0], bounding_box[1]):
        link_com = (bounding_box[0] + bounding_box[1]) / 2

    return link_com, all_pc


def get_link_id_from_name(simulator, object_name, link_name):
    object_id = simulator.urdf_ids[object_name]
    num_joints = p.getNumJoints(object_id, physicsClientId=simulator.id)
    joint_index = None
    for i in range(num_joints):
        joint_info = p.getJointInfo(object_id, i, physicsClientId=simulator.id)
        if joint_info[12].decode("utf-8") == link_name:
            joint_index = i
            break

    return joint_index


def get_joint_id_from_name(simulator, object_name, joint_name):
    object_id = simulator.urdf_ids[object_name]
    num_joints = p.getNumJoints(object_id, physicsClientId=simulator.id)
    joint_index = None
    for i in range(num_joints):
        joint_info = p.getJointInfo(object_id, i, physicsClientId=simulator.id)
        if joint_info[1].decode("utf-8") == joint_name:
            joint_index = i
            break

    return joint_index
