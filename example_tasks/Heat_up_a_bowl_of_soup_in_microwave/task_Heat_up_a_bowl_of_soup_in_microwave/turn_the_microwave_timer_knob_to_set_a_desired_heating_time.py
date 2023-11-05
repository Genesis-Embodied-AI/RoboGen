
from manipulation.sim import SimpleEnv
import numpy as np
from manipulation.gpt_reward_api import *
from manipulation.gpt_primitive_api import *
import gym

class turn_the_microwave_timer_knob_to_set_a_desired_heating_time(SimpleEnv):

    def __init__(self, task_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_name = task_name
        self.detected_position = {}

    def _compute_reward(self):
        # This reward encourages the end-effector to stay near the knob to grasp it.
        eef_pos = get_eef_pos(self)[0]
        knob_pos = get_link_state(self, "Microwave", "link_1")
        reward_near = -np.linalg.norm(eef_pos - knob_pos)
        
        joint_angle = get_joint_state(self, "Microwave", "joint_1") 
        
        joint_limit_low, joint_limit_high = get_joint_limit(self, "Microwave", "joint_1")
        desired_time = joint_limit_high # + (joint_limit_high - joint_limit_low)  / 2 # We assume the target desired time is half of the joint angle. It can also be 1/3, or other values between joint_limit_low and joint_limit_high. 
        
        # The reward is the negative distance between the current joint angle and the joint angle of the desired time.
        diff = np.abs(joint_angle - desired_time)
        reward_joint =  -diff
        
        reward = reward_near + 5 * reward_joint
        
        success = diff < 0.1 * (joint_limit_high - joint_limit_low)
        
        return reward, success

gym.register(
    id='gym-turn_the_microwave_timer_knob_to_set_a_desired_heating_time-v0',
    entry_point=turn_the_microwave_timer_knob_to_set_a_desired_heating_time,
)
