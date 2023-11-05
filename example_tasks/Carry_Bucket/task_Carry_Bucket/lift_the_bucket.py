
from manipulation.sim import SimpleEnv
import numpy as np
from manipulation.gpt_reward_api import *
from manipulation.gpt_primitive_api import *
import gym

class lift_the_bucket(SimpleEnv):

    def __init__(self, task_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_name = task_name
        self.detected_position = {}

    def _compute_reward(self):
        # This reward encourages the end-effector to stay near the handle to grasp it.
        eef_pos = get_eef_pos(self)[0]
        handle_pos = get_link_state(self, "Bucket", "link_0")
        reward_near = -np.linalg.norm(eef_pos - handle_pos)
        
        # The reward is to encourage the robot to lift the bucket. 
        # The goal is to move the bucket to a higher location.
        bucket_pos = get_position(self, "Bucket")
        target_height = 1.0 # We assume the target height is 1.0 meter. It can also be other values.
        diff = np.abs(bucket_pos[2] - target_height)
        reward_height = -diff
        
        reward = reward_near + 5 * reward_height
        success = diff < 0.1 # The task is considered to be successful if the height of the bucket is within 0.1 meter of the target height.
        return reward, success

gym.register(
    id='gym-lift_the_bucket-v0',
    entry_point=lift_the_bucket,
)
