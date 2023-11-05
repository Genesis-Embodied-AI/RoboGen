
from manipulation.sim import SimpleEnv
import numpy as np
from manipulation.gpt_reward_api import *
from manipulation.gpt_primitive_api import *
import gym

class pull_the_lever_to_start_the_brewing_process(SimpleEnv):

    def __init__(self, task_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_name = task_name
        self.detected_position = {}

    def _compute_reward(self):
        # This reward encourages the end-effector to stay near the lever to grasp it.
        eef_pos = get_eef_pos(self)[0]
        lever_pos = get_link_state(self, "CoffeeMachine", "link_6")
        reward_near = -np.linalg.norm(eef_pos - lever_pos)
        
        # Get the joint state of the lever. The semantics and the articulation tree show that joint_6 connects link_6 and is the joint that controls the rotation of the lever.
        joint_angle = get_joint_state(self, "CoffeeMachine", "joint_6") 
        # The reward is the negative distance between the current joint angle and the joint angle when the lever is fully pulled (upper limit).
        joint_limit_low, joint_limit_high = get_joint_limit(self, "CoffeeMachine", "joint_6")
        diff = np.abs(joint_angle - joint_limit_high)
        reward_joint =  -diff
        
        reward = reward_near + 5 * reward_joint
        
        success = diff < 0.1 * (joint_limit_high - joint_limit_low)
        
        return reward, success

gym.register(
    id='gym-pull_the_lever_to_start_the_brewing_process-v0',
    entry_point=pull_the_lever_to_start_the_brewing_process,
)
