
from manipulation.sim import SimpleEnv
import numpy as np
from manipulation.gpt_reward_api import *
from manipulation.gpt_primitive_api import *
import gym

class tilt_the_display_screen_to_adjust_viewing_angle(SimpleEnv):

    def __init__(self, task_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_name = task_name
        self.detected_position = {}

    def _compute_reward(self):
        # This reward encourages the end-effector to stay near the base to grasp it.
        eef_pos = get_eef_pos(self)[0]
        base_pos = get_link_state(self, "Display", "link_1")
        reward_near = -np.linalg.norm(eef_pos - base_pos)
        
        # Get the joint state of the screen. The semantics and the articulation tree show that joint_0 connects link_0 and is the joint that controls the rotation of the screen.
        joint_angle = get_joint_state(self, "Display", "joint_0") 
        # The reward is the negative distance between the current joint angle and the joint angle when the screen is tilted to a desired angle.
        joint_limit_low, joint_limit_high = get_joint_limit(self, "Display", "joint_0")
        desired_angle = joint_limit_low + (joint_limit_high - joint_limit_low) / 2 # We assume the target desired angle is the middle of the joint angle. It can also be 1/3, or other values between joint_limit_low and joint_limit_high. 
        diff = np.abs(joint_angle - desired_angle)
        reward_joint =  -diff
        
        reward = reward_near + 5 * reward_joint
        
        success = diff < 0.1 * (joint_limit_high - joint_limit_low)
        
        return reward, success

gym.register(
    id='gym-tilt_the_display_screen_to_adjust_viewing_angle-v0',
    entry_point=tilt_the_display_screen_to_adjust_viewing_angle,
)
