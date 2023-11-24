from gpt_4.prompts.prompt_manipulation import generate_task as generate_task_manipulation
from gpt_4.prompts.prompt_distractor import generate_distractor
from gpt_4.prompts.prompt_locomotion import generate_task_locomotion
from manipulation.partnet_category import partnet_categories
import json
import time, datetime
import numpy as np
import os
import yaml
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('--train', type=int, default=1)
parser.add_argument('--category', type=str, default=None)
args = parser.parse_args()

temperature_dict = {
    "task_generation": 0.6,
    "reward": 0.2,
    "yaml": 0.3,
    "size": 0.1,
    "joint": 0,
    "spatial_relationship": 0
}

model_dict = {
    "task_generation": "gpt-4",
    "reward": "gpt-4",
    "yaml": "gpt-4",
    "size": "gpt-4",
    "joint": "gpt-4",
    "spatial_relationship": "gpt-4"
}


### store information
### generate task, return config path
meta_path = "generated_tasks_release"
if not os.path.exists("data/{}".format(meta_path)):
    os.makedirs("data/{}".format(meta_path))
time_string = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')
save_folder = "data/{}/meta-temperature-{}.json".format(meta_path, time_string)
with open(save_folder, 'w') as f:
    json.dump(temperature_dict, f, indent=4)
save_folder = "data/{}/meta-model-{}.json".format(meta_path, time_string)
with open(save_folder, 'w') as f:
    json.dump(model_dict, f, indent=4)

np.random.seed(int(time.time()))
gen_func = np.random.choice(['manipulation', 'locomotion'])
gen_func = 'manipulation' if args.category is not None else gen_func
if gen_func == 'manipulation':
    if args.category is None:
        object_category = partnet_categories[np.random.randint(len(partnet_categories))]
    else:
        object_category = args.category
    all_task_config_paths = generate_task_manipulation(object_category, temperature_dict=temperature_dict, model_dict=model_dict, meta_path=meta_path)
    for task_config_path in all_task_config_paths:
        generate_distractor(task_config_path, temperature_dict=temperature_dict, model_dict=model_dict)
elif gen_func == 'locomotion':
    all_task_config_paths = generate_task_locomotion(temperature_dict=temperature_dict, model_dict=model_dict, meta_path=meta_path)
    

if args.train:
    for task_config_path in all_task_config_paths:
        print("trying to learn skill: ", task_config_path)
        try:
            if gen_func == 'manipulation':
                print("task_config_path: ", task_config_path)
                with open(task_config_path, 'r') as f:
                    task_config = yaml.safe_load(f)
                solution_path = None
                for obj in task_config:
                    if "solution_path" in obj:
                        solution_path = obj["solution_path"]
                        break
                
                ## run RL once for each substep
                os.system("python execute.py --task_config_path {}".format(task_config_path))
                ## or run RL multiple times for each substep and pick the best result for next substep
                # os.system("python execute_long_horizon.py --task_config_path {}".format(task_config_path))
            else:
                os.system("python execute_locomotion.py --task_config_path {} & ".format(task_config_path))

        except Exception as e:
            print("=" * 20, "an error occurred", "=" * 20)
            print("an error occurred: ", e)
            print("=" * 20, "an error occurred", "=" * 20)
            print("failed to execute task: ", task_config_path)
            continue
