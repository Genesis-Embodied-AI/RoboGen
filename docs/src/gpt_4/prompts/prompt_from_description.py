from gpt_4.query import query
from gpt_4.prompts.utils import build_task_given_text
from gpt_4.prompts.prompt_distractor import generate_distractor
import time, datetime, os, copy

user_contents = [
"""
I will give you a task name, which is for a robot arm to learn to manipulate an articulated object in household scenarios. I will provide you with the articulated object’s articulation tree and semantics. Your goal is to expand the task description to more information needed for the task. You can think of the robotic arm as a Franka Panda robot. The task will be built in a simulator for the robot to learn it. 

Given a task name, please reply with the following additional information in the following format: 
Description: some basic descriptions of the tasks. 
Additional Objects: Additional objects other than the provided articulated object required for completing the task. If no additional objects are needed, this should be None. 
Links: Links of the articulated objects that are required to perform the task. 
- Link 1: reasons why this link is needed for the task
- Link 2: reasons why this link is needed for the task
- …
Joints: Joints of the articulated objects that are required to perform the task. 
- Joint 1: reasons why this joint is needed for the task
- Joint 2: reasons why this joint is needed for the task
- …


Example Input: 
Task name: Heat a hamburger Inside Oven
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
Description: The robot arm places a hamburger inside the oven, and sets the oven temperature to be appropriate for heating the hamburger.
Additional Objects: hamburger
Links:
- link_0: link_0 is the oven door from the semantics. The robot needs to open the door in order to put the hamburger inside the oven.
link_1: the robot needs to approach link_1, which is the temperature knob, to rotate it to set the desired temperature.
Joints:
- joint_0: from the articulation tree, this is the revolute joint that connects link_0 (the door). Therefore, the robot needs to actuate this joint for opening the door.
- joint_1: from the articulation tree, joint_1 connects link_1, which is the temperature knob. The robot needs to actuate it to rotate link_1 to the desired temperature.

Another example:
Input:
Task name: Retrieve Item from Safe

```Safe articulation tree
links: 
base
link_0
link_1
link_2

joints: 
joint_name: joint_0 joint_type: revolute parent_link: link_2 child_link: link_0
joint_name: joint_1 joint_type: continuous parent_link: link_0 child_link: link_1
joint_name: joint_2 joint_type: fixed parent_link: base child_link: link_2
```

```Safe semantics
link_0 hinge door
link_1 hinge knob
link_2 heavy safe_body
```

Output: 
Description: The robot arm opens the safe, retrieves an item from inside it, and then closes the safe again.
Additional Objects: Item to retrieve from safe.
Links:
- link_0: Link_0 is the safe door from the semantics. The robot needs to open the door in order to retrieve the item from the safe.
- link_1: Link_1 is the safe knob. The robot needs to rotate this knob both to open the safe and to lock it again after retrieving the item.
Joints:
- joint_0: From the articulation tree, this is the revolute joint that connects link_0. The robot needs to actuate this joint to open and close the door.
- joint_1: From the articulation tree, joint_1 connects link_1, which is the safe knob. The robot needs to actuate this joint to rotate link_1 and both unlock and lock the safe.

One more example:
Task Name: Open Door

```Door articulation tree
links: 
base
link_0
link_1
link_2

joints: 
joint_name: joint_0 joint_type: revolute parent_link: link_1 child_link: link_0
joint_name: joint_1 joint_type: fixed parent_link: base child_link: link_1
joint_name: joint_2 joint_type: revolute parent_link: link_0 child_link: link_2
```

```Door semantics
link_0 hinge rotation_door
link_1 static door_frame
link_2 hinge rotation_door
```

Output:
Description: The robotic arm will open the door.
Additional Objects: None
Links:
- link_0: from the semantics, this is the hinge rotation door. The robot needs to approach this link in order to open it. 
Joints: 
- joint_0: from the articulation tree, this is the revolute joint that connects link_0. Therefore, the robot needs to actuate this joint for opening the door.

Can you do the same for the following task and object:
"""
]

def parse_response(task_response):
    task_response = '\n'.join([line for line in task_response.split('\n') if line.strip()])
    task_response = task_response.split('\n')
    task_description = None
    additional_objects = None
    links = None
    joints = None
    for l_idx, line in enumerate(task_response):
        if line.lower().startswith("description:"):
            task_description = task_response[l_idx].split(":")[1].strip()
            task_description = task_description.replace("/", " or ").replace(".", "").replace("'", "").replace('"', "").replace(")", "").replace("(", "")
            additional_objects = task_response[l_idx+1].split(":")[1].strip()
            involved_links = ""
            for link_idx in range(l_idx+3, len(task_response)):
                if task_response[link_idx].lower().startswith("joints:"):
                    break
                else:
                    involved_links += (task_response[link_idx][2:])
            links = involved_links
            involved_joints = ""
            for joint_idx in range(link_idx+1, len(task_response)):
                if not task_response[joint_idx].lower().startswith("- "):
                    break
                else:
                    involved_joints += (task_response[joint_idx][2:])
            joints = involved_joints
            break

    return task_description, additional_objects, links, joints


def expand_task_name(task_name, object_category, object_path, meta_path="generated_task_from_description", temperate=0, model="gpt-4"):
    ts = time.time()
    time_string = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
    save_folder = "data/{}/{}_{}_{}_{}".format(meta_path, task_name.replace(" ", "_"), object_category, object_path, time_string)
    if not os.path.exists(save_folder + "/gpt_response"):
        os.makedirs(save_folder + "/gpt_response")
    
    save_path = "{}/gpt_response/task_generation.json".format(save_folder)

    articulation_tree_path = f"data/dataset/{object_path}/link_and_joint.txt"
    with open(articulation_tree_path, 'r') as f:
        articulation_tree = f.readlines()
    
    semantics = f"data/dataset/{object_path}/semantics.txt"
    with open(semantics, 'r') as f:
        semantics = f.readlines()

    task_user_contents_filled = copy.deepcopy(user_contents[0])
    task_name_filled = "Task name: {}\n".format(task_name)
    articulation_tree_filled = """
```{} articulation tree
{}
```""".format(object_category, "".join(articulation_tree))
    semantics_filled = """
```{} semantics
{}
```""".format(object_category, "".join(semantics))
    task_user_contents_filled = task_user_contents_filled + task_name_filled + articulation_tree_filled + semantics_filled


    system = "You are a helpful assistant."
    task_response = query(system, [task_user_contents_filled], [], save_path=save_path, debug=False, temperature=0, model=model)

    ### parse the response
    task_description, additional_objects, links, joints = parse_response(task_response)    

    return task_description, additional_objects, links, joints, save_folder, articulation_tree_filled, semantics_filled

def generate_from_task_name(task_name, object_category, object_path, temperature_dict=None, model_dict=None, meta_path="generated_task_from_description"):
    expansion_model = model_dict.get("expansion", "gpt-4")
    expansion_temperature = temperature_dict.get("expansion", 0)
    task_description, additional_objects, links, joints, save_folder, articulation_tree_filled, semantics_filled = expand_task_name(
        task_name, object_category, object_path, meta_path, temperate=expansion_temperature, model=expansion_model)
    config_path = build_task_given_text(object_category, task_name, task_description, additional_objects, links, joints, 
                          articulation_tree_filled, semantics_filled, object_path, save_folder, temperature_dict, model_dict)
    return config_path
    
if __name__ == "__main__":
    import argparse
    import numpy as np
    from objaverse_utils.utils import partnet_mobility_dict
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_description', type=str, default="put a pen into the box")
    parser.add_argument('--object', type=str, default="Box")
    parser.add_argument('--object_path', type=str, default="100426")
    args = parser.parse_args()
    
    temperature_dict = {
        "reward": 0,
        "yaml": 0,
        "size": 0,
        "joint": 0,
        "spatial_relationship": 0,
    }
    
    model_dict = {
        "reward": "gpt-4",
        "yaml": "gpt-4",
        "size": "gpt-4",
        "joint": "gpt-4",
        "spatial_relationship": "gpt-4",
    }

    meta_path = "generated_task_from_description"
    assert args.object in partnet_mobility_dict.keys(), "You should use articulated objects in the PartNet Mobility dataset."
    if args.object_path is None:
        possible_object_ids = partnet_mobility_dict[args.object]
        args.object_path = possible_object_ids[np.random.randint(len(possible_object_ids))]
    config_path = generate_from_task_name(args.task_description, args.object, args.object_path, 
        temperature_dict=temperature_dict, meta_path=meta_path, model_dict=model_dict)
    generate_distractor(config_path, temperature_dict=temperature_dict, model_dict=model_dict)
    
    