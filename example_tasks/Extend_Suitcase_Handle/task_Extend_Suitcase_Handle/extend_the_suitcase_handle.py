
from manipulation.sim import SimpleEnv
import numpy as np
from manipulation.gpt_reward_api import *
from manipulation.gpt_primitive_api import *
import gym

class extend_the_suitcase_handle(SimpleEnv):

    def __init__(self, task_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_name = task_name
        self.detected_position = {}

    def _compute_reward(self):
        # This reward encourages the end-effector to stay near the handle to grasp it.
        eef_pos = get_eef_pos(self)[0]
        handle_pos = get_link_state(self, "Suitcase", "link_0")
        reward_near = -np.linalg.norm(eef_pos - handle_pos)
        
        # Get the joint state of the handle. The semantics and the articulation tree show that joint_0 connects link_0 and is the joint that controls the translation of the handle.
        joint_value = get_joint_state(self, "Suitcase", "joint_0") 
        # The reward is the negative distance between the current joint value and the joint value when the handle is fully extended (upper limit).
        joint_limit_low, joint_limit_high = get_joint_limit(self, "Suitcase", "joint_0")
        diff = np.abs(joint_value - joint_limit_high)
        reward_joint =  -diff
        
        reward = reward_near + 5 * reward_joint
        
        success = diff < 0.1 * (joint_limit_high - joint_limit_low)
        
        return reward, success

gym.register(
    id='gym-extend_the_suitcase_handle-v0',
    entry_point=extend_the_suitcase_handle,
)
