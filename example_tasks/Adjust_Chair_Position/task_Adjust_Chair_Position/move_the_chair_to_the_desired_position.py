
from manipulation.sim import SimpleEnv
import numpy as np
from manipulation.gpt_reward_api import *
from manipulation.gpt_primitive_api import *
import gym

class move_the_chair_to_the_desired_position(SimpleEnv):

    def __init__(self, task_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_name = task_name
        self.detected_position = {}

    def _compute_reward(self):
        # This reward encourages the end-effector to stay near the chair to grasp it.
        eef_pos = get_eef_pos(self)[0]
        chair_pos = get_link_state(self, "FoldingChair", "link_0")
        reward_near = -np.linalg.norm(eef_pos - chair_pos)
        
        # The main reward is to encourage the robot to move the chair to the desired position.
        # Here we assume the desired position is at (1.0, 1.0, 0), it can be any position in the environment.
        desired_position = np.array([1.0, 1.0, 0])
        diff = np.linalg.norm(chair_pos - desired_position)
        reward_move = -diff
        
        reward = reward_near + 5 * reward_move
        success = diff < 0.1
        
        return reward, success

gym.register(
    id='gym-move_the_chair_to_the_desired_position-v0',
    entry_point=move_the_chair_to_the_desired_position,
)
