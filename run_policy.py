import yaml
from manipulation.utils import save_numpy_as_gif, build_up_env
from RL.ray_learn import load_policy, make_env  
import numpy as np
import os
import ray
from ray import tune

if not ray.is_initialized():
    ray.init(num_cpus=1, ignore_reinit_error=True, log_to_driver=False)

# build the environment
# NOTE: change to your taks config path
task_config_path = "data/generated_task_from_description/open_the_storage_furniture_StorageFurniture_48452_2023-12-25-16-50-52/open_the_storage_furniture_The_robot_arm_will_open_the_storage_furniture_such_as_a_cabinet_or_a_drawer.yaml"
with open(task_config_path, 'r') as file:
    task_config = yaml.safe_load(file)

solution_path = None
for obj in task_config:
    if "solution_path" in obj:
        solution_path = obj["solution_path"]
        break

# NOTE: change to your task name
task_name = "open_the_door_of_the_storage_furniture"
# NOTE: this is important, this should be set to final state before running the RL algorithm. Change to your state file
last_restore_state_file = "data/generated_task_from_description/open_the_storage_furniture_StorageFurniture_48452_2023-12-25-16-50-52/task_open_the_storage_furniture/primitive_states/2023-12-25-17-05-33/grasp_the_door_of_the_storage_furniture/state_134.pkl" 
obj_id = 0
gui = True
randomize = False

env_config = {
    "task_config_path": task_config_path,
    "solution_path": solution_path,
    "task_name": task_name,
    "last_restore_state_file": last_restore_state_file,
    "action_space": "delta-translation", # NOTE: use the proper action space for the task
    "randomize": randomize,
    "use_bard": True,
    "obj_id": obj_id,
    "use_gpt_size": True,
    "use_gpt_joint_angle": True,
    "use_gpt_spatial_relationship": True,
    "use_distractor": True
}

env = make_env(env_config, render=gui)

env_name = task_name
tune.register_env(env_name, lambda config: make_env(config))


# load the policy
algo = 'sac'
# NOTE: change to your policy path
load_policy_path = "data/generated_task_from_description/open_the_storage_furniture_StorageFurniture_48452_2023-12-25-16-50-52/task_open_the_storage_furniture/RL_sac/2023-12-25-17-05-33/open_the_door_of_the_storage_furniture/best_model/checkpoint_001349/checkpoint-1349"
agent, checkpoint_path = load_policy(algo, env_name, load_policy_path, env_config=env_config, seed=0)

obs = env.reset()
done = False
ret = 0
rgbs = []
state_files = []
states = []
while not done:
    # Compute the next action using the trained policy
    action = agent.compute_action(obs, explore=False)
    print("action: ", action)
    # Step the simulation forward using the action from our trained policy
    obs, reward, done, info = env.step(action)
    ret += reward
    rgb, depth = env.render()
    rgbs.append(rgb)
        
save_numpy_as_gif(np.array(rgbs), "data/eval.gif")

