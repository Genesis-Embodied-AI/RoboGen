
from manipulation.sim import SimpleEnv
import numpy as np
from manipulation.gpt_primitive_api import *
from manipulation.gpt_reward_api import *
import gym

class release_the_toggle_button(SimpleEnv):

    def __init__(self, task_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_name = task_name
        self.detected_position = {}

    def execute(self):
        rgbs, final_state = release_grasp(self)
        success = not check_grasped(self, "Lamp", "link_1")

        return rgbs, final_state, success

gym.register(
    id='release_the_toggle_button-v0',
    entry_point=release_the_toggle_button,
)
