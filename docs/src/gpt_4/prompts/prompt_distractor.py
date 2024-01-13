import json
from gpt_4.query import query
import os
import copy
import yaml
import numpy as np
import torch
import json
from gpt_4.verification import check_text_similarity
from gpt_4.prompts.utils import parse_response_to_get_yaml

user_contents = [
"""
Given a task, which is for a mobile Franka panda robotic arm to learn a manipulation skill in the simulator, your goal is to add more objects into the task scene such that the scene looks more realistic. The Franka panda arm is mounted on a floor, at location (1, 1, 0). It can move freely on the floor. The z axis is the gravity axis. 

The input to you includes the following:
Task name, task description, the essential objects involved in the task, and a config describing the current task scene, which contains only the essential objects needed for the task. The config is a yaml file in the following format:

```yaml 
- use_table: whether the task requires using a table. This should be decided based on common sense. If a table is used, its location will be fixed at (0, 0, 0). The height of the table will be 0.6m. 
# for each object involved in the task, we need to specify the following fields for it.
- type: mesh
  name: name of the object, so it can be referred to in the simulator
  size: describe the scale of the object mesh using 1 number in meters. The scale should match real everyday objects. E.g., an apple is of scale 0.08m. You can think of the scale to be the longest dimension of the object. 
  lang: this should be a language description of the mesh. The language should be a bit detailed, such that the language description can be used to search an existing database of objects to find the object.
  path: this can be a string showing the path to the mesh of the object. 
  on_table: whether the object needs to be placed on the table (if there is a table needed for the task). This should be based on common sense and the requirement of the task.     
  center: the location of the object center. If there isn't a table needed for the task or the object does not need to be on the table, this center should be expressed in the world coordinate system. If there is a table in the task and the object needs to be placed on the table, this center should be expressed in terms of the table coordinate, where (0, 0, 0) is the lower corner of the table, and (1, 1, 1) is the higher corner of the table. In either case, you should try to specify a location such that there is no collision between objects.
```

Your task is to think about what other distractor objects can be added into the scene to make the scene more complex and realistic for the robot to learn the task. These distractor objects are not necessary for the task itself, but their existence makes the scene look more interesting and complex. You should output the distractor objects using the same format as the input yaml file. You should try to put these distractor objects at locations such that they don’t collide with objects already in the scene. 

Here is one example:

Input:

Task name: Heat up a bowl of soup in the microwave
Task description: The robot will grab the soup and move it into the microwave, and then set the temperature to heat it.
Objects involved: Microwave, a bowl of soup
Config:
```yaml
-   use_table: true
-   center: (0.3, 0.7, 0)
    lang: A standard microwave with a turntable and digital timer
    name: Microwave
    on_table: true
    path: microwave.urdf
    size: 0.6
    type: urdf
-   center: (0.2, 0.2, 0)
    lang: A ceramic bowl full of soup
    name: Bowl of Soup
    on_table: true
    path: bowl_soup.obj
    size: 0.15
    type: mesh
```

Output: 
```yaml
- name: plate # a plate is a common object placed when there is microwave and bowl of soup, in a kitchen setup
  lang: a common kitchen plate
  on_table: True
  center: (0.8, 0.8, 0)
  type: mesh
  path: "plate.obj"
  size: 0.15 # a plate is usually of scale 0.15m
- name: sponge # a sponge is a common object placed when there is microwave and bowl of soup, in a kitchen setup
  lang: a common sponge
  on_table: True
  center: (0.5, 0.2, 0)
  type: mesh
  path: "sponge.obj"
  size: 0.1 # a sponge is usually of scale 0.1m
- name: Oven # a oven is a common object placed when there is microwave and bowl of soup, in a kitchen setup
  lang: a kitchen oven
  on_table: False # an oven is usually a standalone object on the floor
  center: (1.8, 0.5, 0) # remember robot is at (1, 1, 0) and table is at (0, 0, 0). So the oven is placed at (1.8, 0.5, 0) in the world coordinate system to avoid collision with other objects.
  type: mesh
  path: "oven.obj"
  size: 0.8 # an oven is usually of scale 0.8m
```

Can you do it for the following task:
"""
]

assistant_contents = [

]
    

def generate_distractor(task_config, temperature_dict, model_dict):
    parent_folder = os.path.dirname(task_config)
    
    existing_response = os.path.join(parent_folder, "gpt_response/task_generation.json")
    
    ori_config = None
    with open(task_config, 'r') as f:
        ori_config = yaml.safe_load(f)
    
    task_name = None
    task_description = None
    for obj in ori_config:
        if "task_name" in obj:
            task_name = obj["task_name"]
        if "task_description" in obj:
            task_description = obj["task_description"]
    task_number = 1
    task_names = [task_name]
    task_descriptions = [task_description]

    input = """
Task name: {}
Task description: {}
Initial config:
```yaml
{}
```
"""
    
    for idx in range(task_number):
        task_name = task_names[idx]
        task_description = task_descriptions[idx]

        copied_config = copy.deepcopy(ori_config)
        new_yaml = []
        for obj in copied_config:
            if "task_description" in obj or "solution_path" in obj or "spatial_relationships" in obj or "set_joint_angle_object_name" in obj:
                continue

            if "uid" in obj:
                del obj["uid"]
            if "all_uid" in obj:
                del obj["all_uid"]
            if "reward_asset_path" in obj.keys():
                del obj["reward_asset_path"]
            
            new_yaml.append(obj)

        description = f"{task_description}".replace(" ", "_").replace(".", "").replace(",", "").replace("(", "").replace(")", "")
        save_name =  description + '.yaml'
        distractor_save_path = os.path.join(parent_folder, save_name.replace(".yaml", "_distractor.yaml"))
        if os.path.exists(distractor_save_path):
            continue

        initial_config = yaml.dump(new_yaml)

        input_filled = copy.deepcopy(input)
        input_filled = input_filled.format(task_name, task_description, initial_config)
        input_filled = user_contents[-1] + input_filled
    
        save_path = os.path.join(parent_folder, "gpt_response/distractor-{}.json".format(task_name.replace(" ", "_")))
        system = "You are a helpful assistant."
        task_response = query(system, [input_filled], [], save_path=save_path, debug=False, temperature=0.2)

        size_save_path = os.path.join(parent_folder, "gpt_response/size_distractor_{}.json".format(task_name))
        response = task_response.split("\n")
        parsed_yaml, _ = parse_response_to_get_yaml(response, task_description, save_path=size_save_path, 
                            temperature=temperature_dict["size"], model=model_dict["size"])

        # some post processing: if the object is close enough to sapian object, we retrieve it from partnet mobility
        sapian_obj_embeddings = torch.load("objaverse_utils/data/partnet_mobility_category_embeddings.pt")
        sapian_object_dict = None
        with open("data/partnet_mobility_dict.json", 'r') as f:
            sapian_object_dict = json.load(f)
        sapian_object_categories = list(sapian_object_dict.keys())
        for obj in parsed_yaml:
            name = obj['name']
            similarity = check_text_similarity(name, check_embeddings=sapian_obj_embeddings)
            max_similarity = np.max(similarity)
            best_category = sapian_object_categories[np.argmax(similarity)]
            if max_similarity > 0.95:
                # retrieve the object from partnet mobility

                obj['type'] = 'urdf'
                obj['name'] = best_category
                object_list = sapian_object_dict[best_category]
                obj['reward_asset_path'] = object_list[np.random.randint(len(object_list))]

        ori_config.append(dict(distractor_config_path=distractor_save_path))

        with open(task_config, 'w') as f:
            yaml.dump(ori_config, f, indent=4)

        with open(distractor_save_path, 'w') as f:
            yaml.dump(parsed_yaml, f, indent=4)


