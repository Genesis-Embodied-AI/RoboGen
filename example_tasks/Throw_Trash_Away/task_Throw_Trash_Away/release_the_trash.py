
from manipulation.sim import SimpleEnv
import numpy as np
from manipulation.gpt_primitive_api import *
from manipulation.gpt_reward_api import *
import gym

class release_the_trash(SimpleEnv):

    def __init__(self, task_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_name = task_name
        self.detected_position = {}

    def execute(self):
        rgbs, final_state = release_grasp(self)
        success = get_grasped_object_name(self) == None

        return rgbs, final_state, success

gym.register(
    id='release_the_trash-v0',
    entry_point=release_the_trash,
)
