
from manipulation.sim import SimpleEnv
import numpy as np
from manipulation.gpt_primitive_api import *
from manipulation.gpt_reward_api import *
import gym

class grasp_the_lid_of_the_trashcan(SimpleEnv):

    def __init__(self, task_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_name = task_name
        self.detected_position = {}

    def execute(self):
        rgbs, final_state = grasp_object_link(self, "TrashCan", "link_2")  
        grasped_object, grasped_link = get_grasped_object_and_link_name(self)
        success = (grasped_object == "TrashCan".lower() and grasped_link == "link_2".lower())

        return rgbs, final_state, success

gym.register(
    id='grasp_the_lid_of_the_trashcan-v0',
    entry_point=grasp_the_lid_of_the_trashcan,
)
