
from manipulation.sim import SimpleEnv
import numpy as np
from manipulation.gpt_reward_api import *
from manipulation.gpt_primitive_api import *
import gym

class flush_the_toilet_by_pushing_the_lever(SimpleEnv):

    def __init__(self, task_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_name = task_name
        self.detected_position = {}

    def _compute_reward(self):
        # This reward encourages the end-effector to stay near the lever to grasp it.
        eef_pos = get_eef_pos(self)[0]
        lever_pos = get_link_state(self, "Toilet", "link_0")
        reward_near = -np.linalg.norm(eef_pos - lever_pos)
        
        # Get the joint state of the lever. The semantics and the articulation tree show that joint_0 connects link_0 and is the joint that controls the rotation of the lever.
        joint_angle = get_joint_state(self, "Toilet", "joint_0") 
        # The reward is the negative distance between the current joint angle and the joint angle when the lever is fully pushed (upper limit).
        joint_limit_low, joint_limit_high = get_joint_limit(self, "Toilet", "joint_0")
        target_joint_angle = joint_limit_high
        
        diff = np.abs(joint_angle - target_joint_angle)
        reward_joint =  -diff
        
        reward = reward_near + 5 * reward_joint
        
        success = diff < 0.1 * (joint_limit_high - joint_limit_low)
        
        return reward, success

gym.register(
    id='gym-flush_the_toilet_by_pushing_the_lever-v0',
    entry_point=flush_the_toilet_by_pushing_the_lever,
)
