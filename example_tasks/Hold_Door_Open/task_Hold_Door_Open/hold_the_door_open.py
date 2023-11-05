
from manipulation.sim import SimpleEnv
import numpy as np
from manipulation.gpt_reward_api import *
from manipulation.gpt_primitive_api import *
import gym

class hold_the_door_open(SimpleEnv):

    def __init__(self, task_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_name = task_name
        self.detected_position = {}

    def _compute_reward(self):
        # This reward encourages the end-effector to stay near the door to hold it.
        eef_pos = get_eef_pos(self)[0]
        door_pos = get_link_state(self, "Door", "link_0")
        reward_near = -np.linalg.norm(eef_pos - door_pos)
        
        # Get the joint state of the door. The semantics and the articulation tree show that joint_0 connects link_0 and is the joint that controls the rotation of the door.
        joint_angle = get_joint_state(self, "Door", "joint_0") 
        # The reward is the negative distance between the current joint angle and the joint angle when the door is fully open (upper limit).
        joint_limit_low, joint_limit_high = get_joint_limit(self, "Door", "joint_0")
        diff = np.abs(joint_angle - joint_limit_high)
        reward_joint =  -diff
        
        reward = reward_near + 5 * reward_joint
        
        success = diff < 0.35 * (joint_limit_high - joint_limit_low)
        
        return reward, success

gym.register(
    id='gym-hold_the_door_open-v0',
    entry_point=hold_the_door_open,
)
