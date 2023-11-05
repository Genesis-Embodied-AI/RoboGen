
from manipulation.sim import SimpleEnv
import numpy as np
from manipulation.gpt_reward_api import *
from manipulation.gpt_primitive_api import *
import gym

class move_the_item_out_of_the_box(SimpleEnv):

    def __init__(self, task_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_name = task_name
        self.detected_position = {}

    def _compute_reward(self):
        # Get the current item position
        item_position = get_position(self, "Item")
        
        # This reward encourages the end-effector to stay near the item
        eef_pos = get_eef_pos(self)[0]
        reward_near = -np.linalg.norm(eef_pos - item_position)
        
        # The reward is to encourage the robot to grasp the item and move the item to be on the table. 
        table_bbox_low, table_bbox_high = get_bounding_box(self, "init_table") # the table is referred to as "init_table" in the simulator. 
        table_bbox_range = table_bbox_high - table_bbox_low
        
        # target location is to put the item at a random location on the table
        target_location = np.zeros(3)
        target_location[0] = table_bbox_low[0] + 0.2 * table_bbox_range[0] # 0.2 is a random chosen number, any number in [0, 1] should work
        target_location[1] = table_bbox_low[1] + 0.3 * table_bbox_range[1] # 0.3 is a random chosen number, any number in [0, 1] should work
        target_location[2] = table_bbox_high[2] # the height should be the table height
        diff = np.linalg.norm(item_position - target_location)
        reward_distance = -diff

        min_aabb, max_aabb = get_bounding_box_link(self, "Box", "link_1") # from the semantics, link_1 is the body of the box.
        reward_out = 0
        if not in_bbox(self, item_position, min_aabb, max_aabb):
            reward_out = 1
        
        reward = reward_near + reward_distance + 2 * reward_out
        
        success = diff < 0.06
        
        return reward, success

gym.register(
    id='gym-move_the_item_out_of_the_box-v0',
    entry_point=move_the_item_out_of_the_box,
)
