import numpy as np
import pybullet as p
from .agent import Agent

class Robot(Agent):
    def __init__(self, controllable_joints, right_arm_joint_indices, right_end_effector, right_gripper_indices):
        self.controllable_joints = controllable_joints 
        self.right_arm_joint_indices = right_arm_joint_indices # Controllable arm joints
        self.controllable_joint_indices = self.right_arm_joint_indices 
        self.right_end_effector = right_end_effector # Used to get the pose of the end effector
        self.right_gripper_indices = right_gripper_indices # Gripper actuated joints
        self.motor_forces = 1.0
        self.motor_gains = 0.05
        super(Robot, self).__init__()

    def init(self, body, id, np_random):
        super(Robot, self).init(body, id, np_random)
        self.joint_max_forces = self.get_joint_max_force(self.controllable_joint_indices)
        self.update_joint_limits()
        self.right_arm_ik_indices = self.right_arm_joint_indices 

    def set_gripper_open_position(self, indices, positions, set_instantly=False, force=500):
        p.setJointMotorControlArray(self.body, jointIndices=indices, controlMode=p.POSITION_CONTROL, targetPositions=positions, positionGains=np.array([0.05]*len(indices)), forces=[force]*len(indices), physicsClientId=self.id)
        if set_instantly:
            self.set_joint_angles(indices, positions, use_limits=True)

    def _is_not_fixed(self, joint_idx):
        joint_info = p.getJointInfo(self.body, joint_idx)
        return joint_info[2] != p.JOINT_FIXED



