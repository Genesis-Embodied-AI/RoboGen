import os

user_contents = [
"""
Your goal is to propose some locomotion tasks for a quadruped/humanoid robot, and writing the corresponding reward functions for the quadruped to learn that specific locomotion skill in a simulator, using reinforcement learning.

Here are some examples:

First example:
Skill: flip rightwards
Reward:
```python
def _compute_reward(self):
    # we first get some information of the quadruped/humanoid robot.
    # COM_pos and COM_quat are the position and orientation (quaternion) of the center of mass of the quadruped/humanoid.
    COM_pos, COM_quat = get_robot_pose(self)
    # COM_vel, COM_ang are the velocity and angular velocity of the center of mass of the quadruped/humanoid.
    COM_vel, COM_ang = get_robot_velocity(self)

    # face_dir, side_dir, and up_dir are three axes of the rotation of the quadruped/humanoid.
    # face direction points from the center of mass towards the face direction of the quadruped/humanoid.
    # side direction points from the center of mass towards the side body direction of the quadruped/humanoid.
    # up direction points from the center of mass towards up, i.e., the negative direction of the gravity. 
    # gravity direction is [0, 0, -1].
    # when initialized, the face of the robot is along the x axis, the side of the robot is along the y axis, and the up of the robot is along the z axis.
    face_dir, side_dir, up_dir = get_robot_direction(self, COM_quat)

    target_side = np.array([0, 1, 0]) # maintain initial side direction during flip
    target_ang = np.array([50, 0, 0.0]) # spin around x axis to do the rightwards flip, since x is the face direction of the robot.

    alpha_ang = 1.0
    alpha_side = 1.0

    r_ang    = - alpha_ang * np.linalg.norm(COM_ang - target_ang)
    r_side   = - alpha_side * np.linalg.norm(side_dir - target_side)
    r += r_ang + r_side

    # there is a default energy term that penalizes the robot for consuming too much energy. This should be included for all skill.
    r_energy = get_energy_reward(self)
    return r + r_energy
```

some more examples:
{}

Can you think of 3 more locomotion skills that a quadruped/humanoid can perform?

For each skill,
Your output format should be:
Skill: <skill name>
Reward:
```python
def _compute_reward(self):
    # your code here
    return r
```
"""
]


good_exmaples = [
"""
Skill: jump backward
Reward:
```python
def _compute_reward(self):
    # we first get some information of the quadruped/humanoid.
    # COM_pos and COM_quat are the position and orientation (quaternion) of the center of mass of the quadruped/humanoid.
    COM_pos, COM_quat = get_robot_pose(self)
    # COM_vel, COM_ang are the velocity and angular velocity of the center of mass of the quadruped/humanoid.
    COM_vel, COM_ang = get_robot_velocity(self)

    # face_dir, side_dir, and up_dir are three axes of the rotation of the quadruped/humanoid.
    # face direction points from the center of mass towards the face direction of the quadruped/humanoid.
    # side direction points from the center of mass towards the side body direction of the quadruped/humanoid.
    # up direction points from the center of mass towards up, i.e., the negative direction of the gravity. 
    # gravity direction is [0, 0, -1].
    # when initialized, the face of the robot is along the x axis, the side of the robot is along the y axis, and the up of the robot is along the z axis.
    face_dir, side_dir, up_dir = get_robot_direction(self, COM_quat)

    if self.time_step <= 30: # first a few steps the robot are jumping
        target_height = 5.0
    else: # then it should not jump
        target_height = 0.0

    target_v = np.array([-5.0, 0, 0.0]) # jump backwards
    target_up = np.array([0, 0, 1]) # maintain up direction
    target_face = np.array([1, 0, 0]) # maintain initial face direction
    target_side = np.array([0, 1, 0]) # maintain initial side direction
    target_ang = np.array([0, 0, 0.0]) # don't let the robot spin

    alpha_vel = 5.0
    alpha_ang = 1.0
    alpha_face = 1.0
    alpha_up = 1.0
    alpha_side = 1.0
    alpha_height = 10.0

    r_vel    = - alpha_vel * np.linalg.norm(COM_vel - target_v)
    r_ang    = - alpha_ang * np.linalg.norm(COM_ang - target_ang)
    r_face   = - alpha_face * np.linalg.norm(face_dir - target_face)
    r_up     = - alpha_up * np.linalg.norm(up_dir - target_up)
    r_side   = - alpha_side * np.linalg.norm(side_dir - target_side)
    r_height = - alpha_height * np.linalg.norm(COM_pos[2] - target_height)
    r = r_vel + r_ang + r_face + r_up + r_side + r_height

    # there is a default energy term that penalizes the robot for consuming too much energy. This should be included for all skill.
    r_energy = get_energy_reward(self)
    return r + r_energy
```
""",

"""
Skill: walk forward
Reward:
```python
def _compute_reward(self):
    # we first get some information of the quadruped/humanoid.
    # COM_pos and COM_quat are the position and orientation (quaternion) of the center of mass of the quadruped/humanoid.
    COM_pos, COM_quat = get_robot_pose(self)
    # COM_vel, COM_ang are the velocity and angular velocity of the center of mass of the quadruped/humanoid.
    COM_vel, COM_ang = get_robot_velocity(self)

    # face_dir, side_dir, and up_dir are three axes of the rotation of the quadruped/humanoid.
    # face direction points from the center of mass towards the face direction of the quadruped/humanoid.
    # side direction points from the center of mass towards the side body direction of the quadruped/humanoid.
    # up direction points from the center of mass towards up, i.e., the negative direction of the gravity. 
    # gravity direction is [0, 0, -1].
    # when initialized, the face of the robot is along the x axis, the side of the robot is along the y axis, and the up of the robot is along the z axis.
    face_dir, side_dir, up_dir = get_robot_direction(self, COM_quat)

    # a skill can be catergorized by target velocity, target body height, target up/side/face direction, as well as target angular velocity of the quadruped/humanoid. 
    target_v = np.array([1.0, 0, 0]) # since the robot faces along x axis initially, for walking forward, the target velocity would just be [1, 0, 0]
    target_height = self.COM_init_pos[2] # we want the robot to keep the original height when walkin, so it does not fall down.
    target_face = np.array([1, 0, 0]) # the target_face keeps the robot facing forward.
    target_side = np.array([0, 1, 0]) # for walking forward, the side direction does not really matter.
    target_up = np.array([0, 0, 1]) # the target_up keeps the robot standing up.
    target_ang = np.array([0, 0, 0]) # for walking forward, the angular velocity does not really matter.

    # note in this example, the real goal can be specified using only 1 term, i.e., the target velocity being [1, 0, 0].
    # howeever, to make the learning reliable, we need the auxiliary terms such as target_height, target_face, and target_up terms to keep the quadruped/humanoid stable during the RL exploration phase.
    # you should try to keep these auxiliary terms for stability as well when desigining the reward.
    
    # we use these coefficients to turn on/off and weight each term. For walking, we only control the target velocity, height, and face and up direction.
    alpha_vel = 1.0
    alpha_height = 1.0
    alpha_face = 1.0
    alpha_side = 0.0
    alpha_up = 1.0
    alpha_ang = 0.0

    r_vel    = - alpha_vel * np.linalg.norm(COM_vel - target_v)
    r_height = - alpha_height * np.linalg.norm(COM_pos[2] - target_height)
    r_face   = - alpha_face * np.linalg.norm(face_dir - target_face)
    r_side    = - alpha_side * np.linalg.norm(side_dir - target_side)
    r_up     = - alpha_up * np.linalg.norm(up_dir - target_up)
    r_ang    = - alpha_ang * np.linalg.norm(COM_ang - target_ang)
    r = r_vel + r_height + r_face + r_side + r_up + r_ang

    # there is a default energy term that penalizes the robot for consuming too much energy. This should be included for all skill.
    r_energy = get_energy_reward(self)
    return r + r_energy
``` 
""", 
]

assistant_contents = []


reward_file_header1 = """
from locomotion.sim import SimpleEnv
import numpy as np
import gym
from locomotion.gpt_reward_api import *

class {}(SimpleEnv):
"""

reward_file_header2 = """
    def __init__(self, task_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_name = task_name
        self.detected_position = {}

"""

reward_file_end = """
gym.register(
    id='gym-{}-v0',
    entry_point={},
)
"""

import time
import datetime
import numpy as np
import copy
from gpt_4.query import query
import yaml
import json

def generate_task_locomotion(model_dict, temperature_dict, meta_path='generated_tasks_locomotion'):
    system = "You are a helpful assistant."
    ts = time.time()
    time_string = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
    save_path = "data/{}/{}".format(meta_path, "locomotion_" + time_string)

    print("=" * 30)
    print("querying GPT to imagine the tasks")
    print("=" * 30)


    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(save_path + "/gpt_response"):
        os.makedirs(save_path + "/gpt_response")

 
    sampled_good_examples = np.random.choice(good_exmaples, 2, replace=False)
    sampled_good_examples = "\n".join(sampled_good_examples)
    new_user_contents = []
    copied_user_contents = copy.deepcopy(user_contents)
    new_user_contents.append(copied_user_contents[0].format(sampled_good_examples))
    task_save_path = os.path.join(save_path, "gpt_response/task_generation.json")
    response = query(system, new_user_contents, assistant_contents, save_path=task_save_path, 
                     model=model_dict['task_generation'], temperature=temperature_dict["task_generation"])

    response = response.split("\n")
    tasks = []
    rewards = []
    
    for l_idx, line in enumerate(response):
        line = line.lower()
        if "skill:" in line.lower():
            tasks.append(response[l_idx])
            reward = []
            start_idx = l_idx + 1
            for l_idx_2 in range(start_idx, len(response)):
                if "```python" in response[l_idx_2].lower():
                    start_idx = l_idx_2 + 1
                    break

            for l_idx_2 in range(start_idx, len(response)):
                if response[l_idx_2].startswith("```"):
                    break
                reward.append(response[l_idx_2])

            reward[0] = "    " + reward[0]
            for idx in range(1, len(reward)):
                reward[idx] = "        " + reward[idx]

            reward = "\n".join(reward)
            rewards.append(reward)
    
    generated_task_configs = []
    for task, reward in zip(tasks, rewards):
        print("task is: ", task)
        task_description = task.split(": ")[1]
        description = f"{task_description}".replace(" ", "_").replace(".", "").replace(",", "").replace("(", "").replace(")", "")

        
        solution_path = os.path.join(save_path, "task_{}".format(description))
        if not os.path.exists(solution_path):
            os.makedirs(solution_path)

        
        reward_file_name = os.path.join(solution_path, f"{description}.py")
        header = reward_file_header1.format(description)
        end = reward_file_end.format(description, description)   
        file_content =  header + reward_file_header2 + reward + end
        with open(reward_file_name, "w") as f:
            f.write(file_content)

        config_path = save_path
        save_name =  description + '.yaml'
        task_config_path = os.path.join(config_path, save_name)
        parsed_yaml = []
        parsed_yaml.append(dict(solution_path=solution_path))
        with open(task_config_path, 'w') as f:
            yaml.dump(parsed_yaml, f, indent=4)
        with open(os.path.join(solution_path, "config.yaml"), 'w') as f:
            yaml.dump(parsed_yaml, f, indent=4)

        generated_task_configs.append(task_config_path)

    return generated_task_configs