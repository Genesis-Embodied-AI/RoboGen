
from manipulation.sim import SimpleEnv
import numpy as np
from manipulation.gpt_primitive_api import *
from manipulation.gpt_reward_api import *
import gym

class grasp_the_object(SimpleEnv):

    def __init__(self, task_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_name = task_name
        self.detected_position = {}

    def execute(self):
        rgbs, final_state = grasp_object(self, "Object")
        success = check_grasped(self, "Object")

        return rgbs, final_state, success

gym.register(
    id='grasp_the_object-v0',
    entry_point=grasp_the_object,
)
