
from manipulation.sim import SimpleEnv
import numpy as np
from manipulation.gpt_reward_api import *
from manipulation.gpt_primitive_api import *
import gym

class move_the_box_out_of_the_cart(SimpleEnv):

    def __init__(self, task_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_name = task_name
        self.detected_position = {}

    def _compute_reward(self):
        # Get the current box position
        box_position = get_position(self, "Box")
        
        # This reward encourages the end-effector to stay near the box to grasp it.
        eef_pos = get_eef_pos(self)[0]
        reward_near = -np.linalg.norm(eef_pos - box_position)
        
        # Get the cart body bounding box
        min_aabb, max_aabb = get_bounding_box_link(self, "Cart", "link_2") # from the semantics, link_2 is the body of the cart.
        diff = np.array(max_aabb) - np.array(min_aabb)
        min_aabb = np.array(min_aabb) + 0.05 * diff  # shrink the bounding box a bit
        max_aabb = np.array(max_aabb) - 0.05 * diff
        
        # another reward is one if the box is outside the cart bounding box
        reward_out = 0
        if not in_bbox(self, box_position, min_aabb, max_aabb): reward_out += 1
        
        # another reward is to encourage the robot to move the box to be away from the cart
        # we need this to give a dense reward signal for the robot to learn to perform this task. 
        reward_reaching = -np.linalg.norm(eef_pos - box_position)
        
        # The task is considered to be successful if the box is outside the cart bounding box
        success = reward_out == 1
        
        # We give more weight to reward_out, which is the major goal of the task.
        reward = 5 * reward_out + reward_reaching + reward_near
        return reward, success

gym.register(
    id='gym-move_the_box_out_of_the_cart-v0',
    entry_point=move_the_box_out_of_the_cart,
)
