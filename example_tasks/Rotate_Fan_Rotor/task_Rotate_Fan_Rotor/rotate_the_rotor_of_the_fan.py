
from manipulation.sim import SimpleEnv
import numpy as np
from manipulation.gpt_reward_api import *
from manipulation.gpt_primitive_api import *
import gym

class rotate_the_rotor_of_the_fan(SimpleEnv):

    def __init__(self, task_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_name = task_name
        self.detected_position = {}

    def _compute_reward(self):
        # This reward encourages the end-effector to stay near the rotor to grasp it.
        eef_pos = get_eef_pos(self)[0]
        rotor_pos = get_link_state(self, "Fan", "link_0")
        reward_near = -np.linalg.norm(eef_pos - rotor_pos)
        
        # Get the joint state of the rotor. The semantics and the articulation tree show that joint_0 connects link_0 and is the joint that controls the rotation of the rotor.
        joint_angle = get_joint_state(self, "Fan", "joint_0") 
        # The reward is the negative distance between the current joint angle and the joint angle when the rotor is fully rotated (upper limit).
        joint_limit_low, joint_limit_high = get_joint_limit(self, "Fan", "joint_0")
        diff = np.abs(joint_angle - joint_limit_high)
        reward_joint =  -diff
        
        reward = reward_near + 5 * reward_joint
        
        success = diff < 0.35 * (joint_limit_high - joint_limit_low)
        
        return reward, success

gym.register(
    id='gym-rotate_the_rotor_of_the_fan-v0',
    entry_point=rotate_the_rotor_of_the_fan,
)
