
from manipulation.sim import SimpleEnv
import numpy as np
from manipulation.gpt_reward_api import *
from manipulation.gpt_primitive_api import *
import gym

class put_the_bowl_of_soup_into_the_microwave(SimpleEnv):

    def __init__(self, task_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_name = task_name
        self.detected_position = {}

    def _compute_reward(self):
        # Get the current bowl position
        bowl_position = get_position(self, "Bowl of soup")
        
        # This reward encourages the end-effector to stay near the bowl to grasp it.
        eef_pos = get_eef_pos(self)[0]
        reward_near = -np.linalg.norm(eef_pos - bowl_position)
        
        # Get the microwave body bounding box
        min_aabb, max_aabb = get_bounding_box_link(self, "Microwave", "link_3") # from the semantics, link_3 is the body of the microwave.
        diff = np.array(max_aabb) - np.array(min_aabb)
        min_aabb = np.array(min_aabb) + 0.05 * diff  # shrink the bounding box a bit
        max_aabb = np.array(max_aabb) - 0.05 * diff
        center = (np.array(max_aabb) + np.array(min_aabb)) / 2
        
        # another reward is one if the bowl is inside the microwave bounding box
        reward_in = 0
        if in_bbox(self, bowl_position, min_aabb, max_aabb): reward_in += 1
        
        # another reward is to encourage the robot to move the bowl to be near the microwave
        reward_reaching = -np.linalg.norm(center - bowl_position)
        
        # The task is considered to be successful if the bowl is inside the microwave bounding box
        success = in_bbox(self, bowl_position, min_aabb, max_aabb)
        
        # We give more weight to reward_in, which is the major goal of the task.
        reward = 5 * reward_in + reward_reaching + reward_near
        return reward, success

gym.register(
    id='gym-put_the_bowl_of_soup_into_the_microwave-v0',
    entry_point=put_the_bowl_of_soup_into_the_microwave,
)
