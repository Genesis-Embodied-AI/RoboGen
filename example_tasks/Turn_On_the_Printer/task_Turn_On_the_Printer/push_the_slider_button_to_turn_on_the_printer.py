
from manipulation.sim import SimpleEnv
import numpy as np
from manipulation.gpt_reward_api import *
from manipulation.gpt_primitive_api import *
import gym

class push_the_slider_button_to_turn_on_the_printer(SimpleEnv):

    def __init__(self, task_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_name = task_name
        self.detected_position = {}

    def _compute_reward(self):
        # This reward encourages the end-effector to stay near the button to grasp it.
        eef_pos = get_eef_pos(self)[0]
        button_pos = get_link_state(self, "Printer", "link_0")
        reward_near = -np.linalg.norm(eef_pos - button_pos)
        
        # Get the joint state of the button. The semantics and the articulation tree show that joint_0 connects link_0 and is the joint that controls the linear motion of the button.
        joint_position = get_joint_state(self, "Printer", "joint_0") 
        # The reward is the negative distance between the current joint position and the joint position when the button is fully pushed (upper limit).
        joint_limit_low, joint_limit_high = get_joint_limit(self, "Printer", "joint_0")
        diff = np.abs(joint_position - joint_limit_high)
        reward_joint =  -diff
        
        reward = reward_near + 5 * reward_joint
        
        success = diff < 0.1 * (joint_limit_high - joint_limit_low)
        
        return reward, success

gym.register(
    id='gym-push_the_slider_button_to_turn_on_the_printer-v0',
    entry_point=push_the_slider_button_to_turn_on_the_printer,
)
