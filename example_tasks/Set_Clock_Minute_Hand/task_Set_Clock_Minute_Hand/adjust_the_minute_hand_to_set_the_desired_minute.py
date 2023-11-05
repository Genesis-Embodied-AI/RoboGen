
from manipulation.sim import SimpleEnv
import numpy as np
from manipulation.gpt_reward_api import *
from manipulation.gpt_primitive_api import *
import gym

class adjust_the_minute_hand_to_set_the_desired_minute(SimpleEnv):

    def __init__(self, task_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_name = task_name
        self.detected_position = {}

    def _compute_reward(self):
        # This reward encourages the end-effector to stay near the minute hand to grasp it.
        eef_pos = get_eef_pos(self)[0]
        hand_pos = get_link_state(self, "Clock", "link_1")
        reward_near = -np.linalg.norm(eef_pos - hand_pos)
        
        # Get the joint state of the minute hand. The semantics and the articulation tree show that joint_1 connects link_1 and is the joint that controls the rotation of the minute hand.
        joint_angle = get_joint_state(self, "Clock", "joint_1") 
        
        joint_limit_low, joint_limit_high = get_joint_limit(self, "Clock", "joint_1")
        desired_minute = joint_limit_low + (joint_limit_high - joint_limit_low)  / 2 # We assume the target desired minute is half of the joint angle. It can also be 1/3, or other values between joint_limit_low and joint_limit_high. 
        
        # The reward is the negative distance between the current joint angle and the joint angle of the desired minute.
        diff = np.abs(joint_angle - desired_minute)
        reward_joint =  -diff
        
        reward = reward_near + 5 * reward_joint
        
        success = diff < 0.1 * (joint_limit_high - joint_limit_low)
        
        return reward, success

gym.register(
    id='gym-adjust_the_minute_hand_to_set_the_desired_minute-v0',
    entry_point=adjust_the_minute_hand_to_set_the_desired_minute,
)
