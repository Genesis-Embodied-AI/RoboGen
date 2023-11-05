
from manipulation.sim import SimpleEnv
import numpy as np
from manipulation.gpt_reward_api import *
from manipulation.gpt_primitive_api import *
import gym

class put_the_object_into_the_drawer(SimpleEnv):

    def __init__(self, task_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_name = task_name
        self.detected_position = {}

    def _compute_reward(self):
        # Get the current object position
        object_position = get_position(self, "Object")
        
        # This reward encourages the end-effector to stay near the object to grasp it.
        eef_pos = get_eef_pos(self)[0]
        reward_near = -np.linalg.norm(eef_pos - object_position)
        
        # Get the drawer bounding box
        min_aabb, max_aabb = get_bounding_box_link(self, "StorageFurniture", "link_2") 
        diff = np.array(max_aabb) - np.array(min_aabb)
        min_aabb = np.array(min_aabb) + 0.05 * diff  # shrink the bounding box a bit
        max_aabb = np.array(max_aabb) - 0.05 * diff
        center = (np.array(max_aabb) + np.array(min_aabb)) / 2
        
        # another reward is one if the object is inside the drawer bounding box
        reward_in = 0
        if in_bbox(self, object_position, min_aabb, max_aabb): reward_in += 1
        
        # another reward is to encourage the robot to move the object to be near the drawer
        reward_reaching = -np.linalg.norm(center - object_position)
        
        # The task is considered to be successful if the object is inside the drawer bounding box
        success = reward_reaching < 0.06
        
        # We give more weight to reward_in, which is the major goal of the task.
        reward = 5 * reward_in + reward_reaching + reward_near
        return reward, success

gym.register(
    id='gym-put_the_object_into_the_drawer-v0',
    entry_point=put_the_object_into_the_drawer,
)
