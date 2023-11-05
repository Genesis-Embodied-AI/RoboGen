
from manipulation.sim import SimpleEnv
import numpy as np
from manipulation.gpt_reward_api import *
from manipulation.gpt_primitive_api import *
import gym

class pull_the_drawer_out(SimpleEnv):

    def __init__(self, task_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_name = task_name
        self.detected_position = {}

    def _compute_reward(self):
        # This reward encourages the end-effector to stay near the drawer to grasp it.
        eef_pos = get_eef_pos(self)[0]
        drawer_pos = get_link_state(self, "Box", "link_1")
        reward_near = -np.linalg.norm(eef_pos - drawer_pos)
        
        # Get the joint state of the drawer. The semantics and the articulation tree show that joint_1 connects link_1 and is the joint that controls the sliding of the drawer.
        joint_value = get_joint_state(self, "Box", "joint_1") 
        # The reward is the negative distance between the current joint value and the joint value when the drawer is fully open (upper limit).
        joint_limit_low, joint_limit_high = get_joint_limit(self, "Box", "joint_1")
        diff = np.abs(joint_value - joint_limit_high)
        reward_joint =  -diff
        
        reward = reward_near + 5 * reward_joint
        
        success = diff < 0.1 * (joint_limit_high - joint_limit_low)
        
        return reward, success

gym.register(
    id='gym-pull_the_drawer_out-v0',
    entry_point=pull_the_drawer_out,
)
