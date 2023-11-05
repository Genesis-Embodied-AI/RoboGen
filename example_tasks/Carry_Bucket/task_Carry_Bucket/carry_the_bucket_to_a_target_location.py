
from manipulation.sim import SimpleEnv
import numpy as np
from manipulation.gpt_reward_api import *
from manipulation.gpt_primitive_api import *
import gym

class carry_the_bucket_to_a_target_location(SimpleEnv):

    def __init__(self, task_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_name = task_name
        self.detected_position = {}

    def _compute_reward(self):
        # This reward encourages the end-effector to stay near the handle to grasp it.
        eef_pos = get_eef_pos(self)[0]
        handle_pos = get_link_state(self, "Bucket", "link_0")
        reward_near = -np.linalg.norm(eef_pos - handle_pos)
        
        # The reward is to encourage the robot to move the bucket to a target location. 
        # The goal is to move the bucket to a target location.
        bucket_pos = get_position(self, "Bucket")
        target_location = np.array([2.0, 2.0, 1.0]) # We assume the target location is at (2.0, 2.0, 1.0). It can also be other locations.
        diff = np.linalg.norm(bucket_pos - target_location)
        reward_distance = -diff
        
        reward = reward_near + 5 * reward_distance
        success = diff < 0.1 # The task is considered to be successful if the bucket is within 0.1 meter of the target location.
        return reward, success

gym.register(
    id='gym-carry_the_bucket_to_a_target_location-v0',
    entry_point=carry_the_bucket_to_a_target_location,
)
