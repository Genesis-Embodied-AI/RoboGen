
from manipulation.sim import SimpleEnv
import numpy as np
from manipulation.gpt_reward_api import *
from manipulation.gpt_primitive_api import *
import gym

class tilt_the_chair_seat_to_a_desired_angle(SimpleEnv):

    def __init__(self, task_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_name = task_name
        self.detected_position = {}

    def _compute_reward(self):
        # This reward encourages the end-effector to stay near the seat to grasp it.
        eef_pos = get_eef_pos(self)[0]
        seat_pos = get_link_state(self, "Chair", "link_11")
        reward_near = -np.linalg.norm(eef_pos - seat_pos)
        
        # Get the joint state of the seat. We know from the semantics and the articulation tree that joint_11 connects link_11 and is the joint that controls the tilt of the seat.
        joint_angle = get_joint_state(self, "Chair", "joint_11") 
        
        joint_limit_low, joint_limit_high = get_joint_limit(self, "Chair", "joint_11")
        desired_tilt = joint_limit_low + (joint_limit_high - joint_limit_low)  / 2 # We assume the target desired tilt is half of the joint angle. It can also be 1/3, or other values between joint_limit_low and joint_limit_high. 
        
        # The reward is the negative distance between the current joint angle and the joint angle of the desired tilt.
        diff = np.abs(joint_angle - desired_tilt)
        reward_joint =  -diff
        reward = reward_near + 5 * reward_joint
        success = diff < 0.1 * (joint_limit_high - joint_limit_low)
        
        return reward, success

gym.register(
    id='gym-tilt_the_chair_seat_to_a_desired_angle-v0',
    entry_point=tilt_the_chair_seat_to_a_desired_angle,
)
