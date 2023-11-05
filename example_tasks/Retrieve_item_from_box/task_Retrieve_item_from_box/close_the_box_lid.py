
from manipulation.sim import SimpleEnv
import numpy as np
from manipulation.gpt_reward_api import *
from manipulation.gpt_primitive_api import *
import gym

class close_the_box_lid(SimpleEnv):

    def __init__(self, task_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_name = task_name
        self.detected_position = {}

    def _compute_reward(self):
        # This reward encourages the end-effector to stay near the lid
        eef_pos = get_eef_pos(self)[0]
        lid_pos = get_link_state(self, "Box", "link_0")
        reward_near = -np.linalg.norm(eef_pos - lid_pos)
        
        # Get the joint state of the lid. The semantics and the articulation tree show that joint_0 connects link_0 and is the joint that controls the rotation of the lid.
        joint_angle = get_joint_state(self, "Box", "joint_0") 
        # The reward encourages the robot to make joint angle of the lid to be the lower limit to close it.
        joint_limit_low, joint_limit_high = get_joint_limit(self, "Box", "joint_0")
        target_joint_angle = joint_limit_low
        
        diff = np.abs(target_joint_angle - joint_angle)
        reward_joint =  -diff
        
        reward = reward_near + 5 * reward_joint
        
        success = diff < 0.1 * (joint_limit_high - joint_limit_low)       
        
        return reward, success

gym.register(
    id='gym-close_the_box_lid-v0',
    entry_point=close_the_box_lid,
)
