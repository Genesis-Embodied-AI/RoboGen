import pybullet as p
import numpy as np
import os

def get_robot_pose(simulator):
    COM_pos, COM_quat = p.getBasePositionAndOrientation(simulator.robot_id, physicsClientId=simulator.id)
    return COM_pos, COM_quat

def get_robot_velocity(simulator):
    COM_vel, COM_ang = p.getBaseVelocity(simulator.robot_id, physicsClientId=simulator.id)
    return COM_vel, COM_ang

def get_robot_direction(simulator, COM_quat):
    face_dir = np.array(p.getMatrixFromQuaternion(COM_quat)).reshape((3, 3))[:, 0]
    side_dir = np.array(p.getMatrixFromQuaternion(COM_quat)).reshape((3, 3))[:, 1]
    up_dir = np.array(p.getMatrixFromQuaternion(COM_quat)).reshape((3, 3))[:, 2]
    return face_dir, side_dir, up_dir

def get_energy_reward(simulator):
    alpha_energy = 1e-6
    r_energy = 0.0
    for joint_id in simulator.joint_ids:
        joint_state = p.getJointState(simulator.robot_id, joint_id, physicsClientId=simulator.id)
        joint_vel = joint_state[1]
        joint_tau = joint_state[3]
        r_energy += - np.power(joint_vel * joint_tau, 2)
    r_energy *= alpha_energy

    return r_energy

    