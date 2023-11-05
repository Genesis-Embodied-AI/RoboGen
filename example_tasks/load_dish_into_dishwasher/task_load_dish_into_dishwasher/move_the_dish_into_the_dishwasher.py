
from manipulation.sim import SimpleEnv
import numpy as np
from manipulation.gpt_reward_api import *
from manipulation.gpt_primitive_api import *
import gym

class move_the_dish_into_the_dishwasher(SimpleEnv):

    def __init__(self, task_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_name = task_name
        self.detected_position = {}

    def _compute_reward(self):
        # Get the current dish position
        dish_position = get_position(self, "Dish")
        
        # This reward encourages the end-effector to stay near the dish to grasp it.
        eef_pos = get_eef_pos(self)[0]
        reward_near = -np.linalg.norm(eef_pos - dish_position)
        
        # Get the dishwasher body bounding box
        min_aabb, max_aabb = get_bounding_box_link(self, "Dishwasher", "link_0") # from the semantics, link_0 is the body of the dishwasher.
        diff = np.array(max_aabb) - np.array(min_aabb)
        min_aabb = np.array(min_aabb) + 0.05 * diff  # shrink the bounding box a bit
        max_aabb = np.array(max_aabb) - 0.05 * diff
        
        # another reward is one if the dish is inside the dishwasher bounding box
        reward_in = 0
        if in_bbox(self, dish_position, min_aabb, max_aabb): reward_in += 1
        
        # another reward is to encourage the robot to move the dish to be near the dishwasher
        # we need this to give a dense reward signal for the robot to learn to perform this task. 
        reward_reaching = -np.linalg.norm(dish_position - np.array(min_aabb))
        
        # The task is considered to be successful if the dish is inside the dishwasher bounding box
        success = in_bbox(self, dish_position, min_aabb, max_aabb)
        
        # We give more weight to reward_in, which is the major goal of the task.
        reward = 5 * reward_in + reward_reaching + reward_near
        return reward, success

gym.register(
    id='gym-move_the_dish_into_the_dishwasher-v0',
    entry_point=move_the_dish_into_the_dishwasher,
)
