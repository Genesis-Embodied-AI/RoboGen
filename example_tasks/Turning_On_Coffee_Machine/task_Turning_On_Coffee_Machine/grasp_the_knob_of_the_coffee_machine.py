
from manipulation.sim import SimpleEnv
import numpy as np
from manipulation.gpt_primitive_api import *
from manipulation.gpt_reward_api import *
import gym

class grasp_the_knob_of_the_coffee_machine(SimpleEnv):

    def __init__(self, task_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_name = task_name
        self.detected_position = {}

    def execute(self):
        rgbs, final_state = grasp_object_link(self, "CoffeeMachine", "link_0")  
        success = check_grasped(self, "CoffeeMachine", "link_0")

        return rgbs, final_state, success

gym.register(
    id='grasp_the_knob_of_the_coffee_machine-v0',
    entry_point=grasp_the_knob_of_the_coffee_machine,
)
