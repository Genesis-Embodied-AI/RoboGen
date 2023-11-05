
from locomotion.sim import SimpleEnv
import numpy as np
import gym
from locomotion.gpt_reward_api import *

class Turn_right(SimpleEnv):

    def __init__(self, task_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_name = task_name
        self.detected_position = {}

    def _compute_reward(self):
            COM_pos, COM_quat = get_robot_pose(self)
            COM_vel, COM_ang = get_robot_velocity(self)
            face_dir, side_dir, up_dir = get_robot_direction(self, COM_quat)
        
            target_ang = np.array([0, 0, -1.0]) # The robot should rotate to the right along the z-axis
            target_v = np.array([0, 0, 0]) # The robot should stay in place while turning
            target_height = self.COM_init_pos[2] # The robot should maintain its height while turning
            target_face = np.array([1, 0, 0]) # The robot should maintain its face direction
            target_side = np.array([0, 1, 0]) # The robot should maintain its side direction
            target_up = np.array([0, 0, 1]) # The robot should maintain its up direction
        
            alpha_vel = 1.0
            alpha_height = 1.0
            alpha_face = 1.0
            alpha_side = 1.0
            alpha_up = 1.0
            alpha_ang = 5.0 # We put more weight on the angular velocity as it is the main goal
        
            r_vel    = - alpha_vel * np.linalg.norm(COM_vel - target_v)
            r_height = - alpha_height * np.linalg.norm(COM_pos[2] - target_height)
            r_face   = - alpha_face * np.linalg.norm(face_dir - target_face)
            r_side   = - alpha_side * np.linalg.norm(side_dir - target_side)
            r_up     = - alpha_up * np.linalg.norm(up_dir - target_up)
            r_ang    = - alpha_ang * np.linalg.norm(COM_ang - target_ang)
            r = r_vel + r_height + r_face + r_side + r_up + r_ang
        
            r_energy = get_energy_reward(self)
            return r + r_energy
gym.register(
    id='gym-Turn_right-v0',
    entry_point=Turn_right,
)
