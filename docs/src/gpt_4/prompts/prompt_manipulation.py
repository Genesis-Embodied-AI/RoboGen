import numpy as np
import copy
import time, datetime
import os
import json
from objaverse_utils.utils import partnet_mobility_dict
from gpt_4.prompts.utils import build_task_given_text, parse_task_response
from gpt_4.query import query

task_user_contents = """
I will give you an articulated object, with its articulation tree and semantics. Your goal is to imagine some tasks that a robotic arm can perform with this articulated object in household scenarios. You can think of the robotic arm as a Franka Panda robot. The task will be built in a simulator for the robot to learn it. 

Focus on manipulation or interaction with the object itself. Sometimes the object will have functions, e.g., a microwave can be used to heat food, in these cases, feel free to include other objects that are needed for the task. 
Please do not think of tasks that try to assemble or disassemble the object. Do not think of tasks that aim to clean the object or check its functionality. 

For each task you imagined, please write in the following format: 
Task name: the name of the task.
Description: some basic descriptions of the tasks. 
Additional Objects: Additional objects other than the provided articulated object required for completing the task. 
Links: Links of the articulated objects that are required to perform the task. 
- Link 1: reasons why this link is needed for the task
- Link 2: reasons why this link is needed for the task
- …
Joints: Joints of the articulated objects that are required to perform the task. 
- Joint 1: reasons why this joint is needed for the task
- Joint 2: reasons why this joint is needed for the task
- …


Example Input: 

```Oven articulation tree
links: 
base
link_0
link_1
link_2
link_3
link_4
link_5
link_6
link_7

joints: 
joint_name: joint_0 joint_type: revolute parent_link: link_7 child_link: link_0
joint_name: joint_1 joint_type: continuous parent_link: link_7 child_link: link_1
joint_name: joint_2 joint_type: continuous parent_link: link_7 child_link: link_2
joint_name: joint_3 joint_type: continuous parent_link: link_7 child_link: link_3
joint_name: joint_4 joint_type: continuous parent_link: link_7 child_link: link_4
joint_name: joint_5 joint_type: continuous parent_link: link_7 child_link: link_5
joint_name: joint_6 joint_type: continuous parent_link: link_7 child_link: link_6
joint_name: joint_7 joint_type: fixed parent_link: base child_link: link_7
```

```Oven semantics
link_0 hinge door
link_1 hinge knob
link_2 hinge knob
link_3 hinge knob
link_4 hinge knob
link_5 hinge knob
link_6 hinge knob
link_7 heavy oven_body
```

Example output:

Task Name: Open Oven Door
Description: The robotic arm will open the oven door.
Additional Objects: None
Links:
- link_0: from the semantics, this is the door of the oven. The robot needs to approach this door in order to open it. 
Joints: 
- joint_0: from the articulation tree, this is the revolute joint that connects link_0. Therefore, the robot needs to actuate this joint for opening the door.


Task Name: Adjust Oven Temperature
Description: The robotic arm will turn one of the oven's hinge knobs to set a desired temperature.
Additional Objects: None
Links:
- link_1: the robot needs to approach link_1, which is assumed to be the temperature knob, to rotate it to set the temperature.
Joints:
- joint_1: joint_1 connects link_1 from the articulation tree. The robot needs to actuate it to rotate link_1 to the desired temperature.


Task Name: Heat a hamburger Inside Oven 
Description: The robot arm places a hamburger inside the oven, and sets the oven temperature to be appropriate for heating the hamburger.
Additional Objects: hamburger
Links:
- link_0: link_0 is the oven door from the semantics. The robot needs to open the door in order to put the hamburger inside the oven.
link_1: the robot needs to approach link_1, which is the temperature knob, to rotate it to set the desired temperature.
Joints:
- joint_0: from the articulation tree, this is the revolute joint that connects link_0 (the door). Therefore, the robot needs to actuate this joint for opening the door.
- joint_1: from the articulation tree, joint_1 connects link_1, which is the temperature knob. The robot needs to actuate it to rotate link_1 to the desired temperature.

Task Name: Set Oven Timer
Description: The robot arm turns a timer knob to set cooking time for the food.
Additional Objects: None.
Links: 
- link_2: link_2 is assumed to be the knob for controlling the cooking time. The robot needs to approach link_2 to set the cooking time.
Joints:
- joint_2: from the articulation tree, joint_2 connects link_2. The robot needs to actuate joint_2 to rotate link_2 to the desired position, setting the oven timer.


Can you do the same for the following object:
"""

# TODO: add another example where the ambiguous description is changed to be a precise description of the object. 

def generate_task(object_category=None, object_path=None, existing_response=None, temperature_dict=None, 
                  model_dict=None, meta_path="generated_tasks"):
    # send the object articulation tree, semantics file and get task descriptions, invovled objects and joints
    # randomly sample an object for generation. 

    object_cetegories = list(partnet_mobility_dict.keys())
    if object_category is None:
        object_category = object_cetegories[np.random.randint(len(object_cetegories))]
    if object_path is None:
        possible_object_ids = partnet_mobility_dict[object_category]
        object_path = possible_object_ids[np.random.randint(len(possible_object_ids))]
    
    articulation_tree_path = f"data/dataset/{object_path}/link_and_joint.txt"
    with open(articulation_tree_path, 'r') as f:
        articulation_tree = f.readlines()
    
    semantics = f"data/dataset/{object_path}/semantics.txt"
    with open(semantics, 'r') as f:
        semantics = f.readlines()

    task_user_contents_filled = copy.deepcopy(task_user_contents)
    articulation_tree_filled = """
```{} articulation tree
{}
```""".format(object_category, "".join(articulation_tree))
    semantics_filled = """
```{} semantics
{}
```""".format(object_category, "".join(semantics))
    task_user_contents_filled = task_user_contents_filled + articulation_tree_filled + semantics_filled

    if existing_response is None:
        system = "You are a helpful assistant."
        ts = time.time()
        time_string = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
        save_folder = "data/{}/{}_{}_{}".format(meta_path, object_category, object_path, time_string)
        if not os.path.exists(save_folder + "/gpt_response"):
            os.makedirs(save_folder + "/gpt_response")

        save_path = "{}/gpt_response/task_generation.json".format(save_folder)

        print("=" * 50)
        print("=" * 20, "generating task", "=" * 20)
        print("=" * 50)

        task_response = query(system, [task_user_contents_filled], [], save_path=save_path, debug=False, 
                              temperature=temperature_dict['task_generation'],
                              model=model_dict['task_generation'])
   
    else:
        with open(existing_response, 'r') as f:
            data = json.load(f)
        task_response = data["res"]
        print(task_response)
            
    ### generate task yaml config
    task_names, task_descriptions, additional_objects, links, joints = parse_task_response(task_response)
    task_number = len(task_names)
    print("task number: ", task_number)

    all_config_paths = []
    for task_idx in range(task_number):
        if existing_response is None:
            time.sleep(20)
        task_name = task_names[task_idx]
        task_description = task_descriptions[task_idx]
        additional_object = additional_objects[task_idx]
        involved_links = links[task_idx]
        involved_joints = joints[task_idx]

        config_path = build_task_given_text(object_category, task_name, task_description, additional_object, involved_links, involved_joints, 
                          articulation_tree_filled, semantics_filled, object_path, save_folder, temperature_dict, model_dict=model_dict)
        all_config_paths.append(config_path)

    return all_config_paths


