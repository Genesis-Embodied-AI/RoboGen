
from manipulation.sim import SimpleEnv
import numpy as np
from manipulation.gpt_reward_api import *
from manipulation.gpt_primitive_api import *
import gym

class move_the_food_into_the_refrigerator(SimpleEnv):

    def __init__(self, task_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_name = task_name
        self.detected_position = {}

    def _compute_reward(self):
        # Get the current food position
        food_position = get_position(self, "Food")
        
        # The first reward encourages the end-effector to stay near the food
        eef_pos = get_eef_pos(self)[0]
        reward_near = -np.linalg.norm(eef_pos - food_position)
        
        # Get the refrigerator body bounding box
        min_aabb, max_aabb = get_bounding_box_link(self, "Refrigerator", "link_2") # from the semantics, link_2 is the body of the refrigerator.
        diff = np.array(max_aabb) - np.array(min_aabb)
        min_aabb = np.array(min_aabb) + 0.05 * diff  # shrink the bounding box a bit
        max_aabb = np.array(max_aabb) - 0.05 * diff
        
        # The reward is to encourage the robot to grasp the food and move the food to be inside the refrigerator. 
        reward_in = 0
        if in_bbox(self, food_position, min_aabb, max_aabb): reward_in += 1
        
        # another reward is to encourage the robot to move the food to be near the refrigerator
        # we need this to give a dense reward signal for the robot to learn to perform this task. 
        center = (np.array(max_aabb) + np.array(min_aabb)) / 2
        reward_reaching = -np.linalg.norm(center - food_position)
        
        # The task is considered to be successful if the food is inside the refrigerator bounding box
        success = in_bbox(self, food_position, min_aabb, max_aabb)
        
        # We give more weight to reward_in, which is the major goal of the task.
        reward = 5 * reward_in + reward_reaching + reward_near
        return reward, success

gym.register(
    id='gym-move_the_food_into_the_refrigerator-v0',
    entry_point=move_the_food_into_the_refrigerator,
)
