
from manipulation.sim import SimpleEnv
import numpy as np
from manipulation.gpt_primitive_api import *
from manipulation.gpt_reward_api import *
import gym

class approach_the_toilet_lever(SimpleEnv):

    def __init__(self, task_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_name = task_name
        self.detected_position = {}

    def execute(self):
        rgbs, final_state = approach_object_link(self, "Toilet", "link_0")  
        success = gripper_close_to_object_link(self, "Toilet", "link_0")

        return rgbs, final_state, success

gym.register(
    id='approach_the_toilet_lever-v0',
    entry_point=approach_the_toilet_lever,
)
