import copy
from gpt_4.query import query
import os

user_contents = [
"""
A robotic arm is trying to solve some household object manipulation tasks to learn corresponding skills in a simulator.

We will provide with you the task description, the initial scene configurations of the task, which contains the objects in the task and certain information about them. 
Your goal is to decompose the task into executable sub-steps for the robot, and for each substep, you should either call a primitive action that the robot can execute, or design a reward function for the robot to learn, to complete the substep. 
For each substep, you should also write a function that checks whether the substep has been successfully completed. 

Common substeps include moving towards a location, grasping an object, and interacting with the joint of an articulated object.

An example task:
Task Name: Set oven temperature
Description: The robotic arm will turn the knob of an oven to set a desired temperature.
Initial config:
```yaml
-   use_table: false
-   center: (1, 0, 0) # when an object is not on the table, the center specifies its location in the world coordinate. 
    lang: a freestanding oven 
    name: oven
    on_table: false
    path: oven.urdf
    size: 0.85
    type: urdf
```

I will also give you the articulation tree and semantics file of the articulated object in the task. Such information will be useful for writing the reward function/the primitive actions, for example, when the reward requires accessing the joint value of a joint in the articulated object, or the position of a link in the articulated object, or when the primitive needs to access a name of the object.
```Oven articulation tree:
links: 
base
link_0
link_1
link_2
link_3
link_4

joints: 
joint_name: joint_0 joint_type: continuous parent_link: link_4 child_link: link_0
joint_name: joint_1 joint_type: continuous parent_link: link_4 child_link: link_1
joint_name: joint_2 joint_type: continuous parent_link: link_4 child_link: link_2
joint_name: joint_3 joint_type: continuous parent_link: link_4 child_link: link_3
joint_name: joint_4 joint_type: fixed parent_link: base child_link: link_4
```

```Oven semantics
link_0 hinge knob
link_1 hinge knob
link_2 hinge knob
link_3 hinge knob
link_4 heavy oven_body
```


I will also give you the links and joints of the articulated object that will be used for completing the task:
Links:
link_0: We know from the semantics that link_0 is a hinge knob. It is assumed to be the knob that controls the temperature of the oven. The robot needs to actuate this knob to set the temperature of the oven.

Joints:
joint_0: from the articulation tree, joint_0 connects link_0 and is a continuous joint. Therefore, the robot needs to actuate joint_0 to turn link_0, which is the knob.


For each substep, you should decide whether the substep can be achieved by using the provided list of primitives. If not, you should then write a reward function for the robot to learn to perform this substep. 
If you choose to write a reward function for the substep, you should also specify the action space of the robot when learning this reward function. 
There are 2 options for the action space: "delta-translation", where the action is the delta translation of the robot end-effector, suited for local movements; and "normalized-direct-translation", where the action specifies the target location the robot should move to, suited for moving to a target location.
For each substep, you should also write a condition that checks whether the substep has been successfully completed.

Here is a list of primitives the robot can do. The robot is equipped with a suction gripper, which makes it easy for the robot to grasp an object or a link on an object. 
grasp_object(self, object_name): the robot arm will grasp the object specified by the argument object name.
grasp_object_link(self, object_name, link_name): some object like an articulated object is composed of multiple links. The robot will grasp a link with link_name on the object with object_name. 
release_grasp(self): the robot will release the grasped object.
Note that all primitives will return a tuple (rgbs, final_state) which represents the rgb images of the execution process and the final state of the execution process. 
You should always call the primitive in the following format:
rgbs, final_state = some_primitive_function(self, arg1, ..., argn)

Here is a list of helper functions that you can use for designing the reward function or the success condition:
get_position(self, object_name): get the position of center of mass of object with object_name.
get_orientation(self, object_name): get the orientation of an object with object_name.
get_joint_state(self, object_name, joint_name): get the joint angle value of a joint in an object.
get_joint_limit(self, object_name, joint_name): get the lower and upper joint angle limit of a joint in an object, returned as a 2-element tuple.
get_link_state(self, object_name, link_name): get the position of the center of mass of the link of an object.
get_eef_pos(self): returns the position, orientation of the robot end-effector as a list.
get_bounding_box(self, object_name): get the axis-aligned bounding box of an object. It returns the min and max xyz coordinate of the bounding box.
get_bounding_box_link(self, object_name, link_name): get the axis-aligned bounding box of the link of an object. It returns the min and max xyz coordinate of the bounding box.
in_bbox(self, pos, bbox_min, bbox_max): check if pos is within the bounding box with the lowest corner at bbox_min and the highest corner at bbox_max. 
check_grasped(self, object_name, link_name): return true if an object or a link of the object is grasped. link_name can be none, in which case it will check whether the object is grasped.
get_initial_pos_orient(self, obj): get the initial position and orientation of an object at the beginning of the task.
get_initial_joint_angle(self, obj_name, joint_name): get the initial joint angle of an object at the beginning of the task.

You can assume that for objects, the lower joint limit corresponds to their natural state, e.g., a box is closed with the lid joint being 0, and a lever is unpushed when the joint angle is 0.

For the above task "Set oven temperature", it can be decomposed into the following substeps, primitives, and reward functions:

substep 1: grasp the temperature knob
```primitive
	rgbs, final_state = grasp_object_link(self, "oven", "link_0") 
    success = check_grasped(self, "oven", "link_0")
```

substep 2: turn the temperature knob to set a desired temperature
```reward
def _compute_reward(self):
    # This reward encourages the end-effector to stay near the knob to grasp it.
    eef_pos = get_eef_pos(self)[0]
    knob_pos = get_link_state(self, "oven", "link_0")
    reward_near = -np.linalg.norm(eef_pos - knob_pos)

    joint_angle = get_joint_state(self, "oven", "joint_0") 
    
    joint_limit_low, joint_limit_high = get_joint_limit(self, "oven", "joint_0")
    desired_temperature = joint_limit_low + (joint_limit_high - joint_limit_low)  / 3 # We assume the target desired temperature is one third of the joint angle. It can also be 1/3, or other values between joint_limit_low and joint_limit_high. 

    # The reward is the negative distance between the current joint angle and the joint angle of the desired temperature.
    diff = np.abs(joint_angle - desired_temperature)
    reward_joint =  -diff
    reward = reward_near + 5 * reward_joint
    success = diff < 0.1 * (joint_limit_high - joint_limit_low)

    return reward, success
```

```action space
delta-translation
```

I will give some more examples of decomposing the task. Reply yes if you understand the goal.
""",

"""
Another example:
Task Name: Fetch item from refrigerator
Description: The robotic arm will open a refrigerator door reach inside to grab an item, place it on the table, and then close the door
Initial config:
```yaml
-   use_table: true 
-   center: (1.2, 0, 0)
    lang: a common two-door refrigerator
    name: Refrigerator
    on_table: false 
    path: refrigerator.urdf
    size: 1.8
    type: urdf
-   center: (1.2, 0, 0.5) 
    lang: a can of soda
    name: Item
    on_table: false 
    path: soda_can.obj
    size: 0.2
    type: mesh
```

```Refrigerator articulation tree
links: 
base
link_0
link_1
link_2

joints: 
joint_name: joint_0 joint_type: fixed parent_link: base child_link: link_0
joint_name: joint_1 joint_type: revolute parent_link: link_0 child_link: link_1
joint_name: joint_2 joint_type: revolute parent_link: link_0 child_link: link_2
```

```Refrigerator semantics
link_0 heavy refrigerator_body
link_1 hinge door
link_2 hinge door
```

Links:
link_1: This link is one of the refrigerator doors, which the robot neesd to reach for the item inside.
Joints:
joint_1: This joint connects link_1, representing one of the doors. The robot needs to actuate this joint to open the door, reach for the item, and close the door.

This task can be decomposed as follows:

substep 1: grasp the refrigerator door
```primitive
    rgbs, final_state = grasp_object_link(self, "Refrigerator", "link_1")  
    success = check_grasped(self, "Refrigerator", "link_1")
```

substep 2: open the refrigerator door
```reward
def _compute_reward(self):
    # this reward encourages the end-effector to stay near door to grasp it.
    eef_pos = get_eef_pos(self)[0]
    door_pos = get_link_state(self, "Refrigerator", "link_1")
    reward_near = -np.linalg.norm(eef_pos - door_pos)

    # Get the joint state of the door. We know from the semantics and the articulation tree that joint_1 connects link_1 and is the joint that controls the rotation of the door.
    joint_angle = get_joint_state(self, "Refrigerator", "joint_1") 
    # The reward is the negative distance between the current joint angle and the joint angle when the door is fully open (upper limit).
    joint_limit_low, joint_limit_high = get_joint_limit(self, "Refrigerator", "joint_1")
    diff = np.abs(joint_angle - joint_limit_high)
    reward_joint =  -diff

    reward = reward_near + 5 * reward_joint
    success = diff < 0.35 * (joint_limit_high - joint_limit_low) # for opening, we think 65 percent is enough

    return reward, success
```

```action space
delta-translation
```
In the last substep the robot already grasps the door, thus only local movements are needed to open it. 

substep 3: grasp the item
```primitive
    rgbs, final_state = grasp_object(self, "Item")
    success = check_grasped(self, "Item")
```

substep 4: move the item out of the refrigerator
```reward
def _compute_reward(self):
    # Get the current item position
    item_pos = get_position(self, "Item")

    # The first reward encourages the end-effector to stay near the item
    eef_pos = get_eef_pos(self)[0]
    reward_near = -np.linalg.norm(eef_pos - item_pos)

    # The reward is to encourage the robot to grasp the item and move the item to be on the table. 
    # The goal is not to just move the soda can to be at a random location out of the refrigerator. Instead, we need to place it somewhere on the table. 
    # This is important for moving an object out of a container style of task.
    table_bbox_low, table_bbox_high = get_bounding_box(self, "init_table") # the table is referred to as "init_table" in the simulator. 
    table_bbox_range = table_bbox_high - table_bbox_low

    # target location is to put the item at a random location on the table
    target_location = np.zeros(3)
    target_location[0] = table_bbox_low[0] + 0.2 * table_bbox_range[0] # 0.2 is a random chosen number, any number in [0, 1] should work
    target_location[1] = table_bbox_low[1] + 0.3 * table_bbox_range[1] # 0.3 is a random chosen number, any number in [0, 1] should work
    target_location[2] = table_bbox_high[2] + 0.05 # target height is slightly above the table
    diff = np.linalg.norm(item_pos - target_location)
    reward_distance = -diff

    reward = reward_near + 5 * reward_distance

    success = diff < 0.06
    
    return reward, success
```

```action space
normalized-direct-translation
```
Since this substep requires moving the item to a target location, we use the normalized-direct-translation.

substep 5: grasp the refrigerator door again
```primitive
    rgbs, final_state = grasp_object_link(self, "Refrigerator", "link_1")
    success = check_grasped(self, "Refrigerator", "link_1") 
```

substep 6: close the refrigerator door
```reward
def _compute_reward(self):
    # this reward encourages the end-effector to stay near door
    eef_pos = get_eef_pos(self)[0]
    door_pos = get_link_state(self, "Refrigerator", "link_1")
    reward_near = -np.linalg.norm(eef_pos - door_pos)

    # Get the joint state of the door. 
    joint_angle = get_joint_state(self, "Refrigerator", "joint_1") 
    # The reward encourages the robot to make joint angle of the door to be the lower limit to clost it.
    joint_limit_low, joint_limit_high = get_joint_limit(self, "Refrigerator", "joint_1")
    diff = np.abs(joint_limit_low - joint_angle)
    reward_joint =  -diff

    reward = reward_near + 5 * reward_joint

    success = diff < 0.1 * (joint_limit_high - joint_limit_low) # for closing, we think 10 percent is enough     

    return reward, success
```

```action space
delta-translation
```

I will provide more examples in the following messages. Please reply yes if you understand the goal.
""",

"""
Here is another example:

Task Name:  Put a toy car inside a box
Description: The robotic arm will open a box, grasp the toy car and put it inside the box.
Initial config:
```yaml
-  use_table: True 
-   center: (0.2, 0.3, 0)
    on_table: True
    lang: a box
    name: box
    size: 0.25
    type: urdf
-   center: (0.1, 0.6, 0)
    on_table: True
    lang: a toy car
    name: toy_car
    size: 0.1
    type: mesh
```

```box articulation tree
links: 
base
link_0
link_1
link_2

joints: 
joint_name: joint_0 joint_type: revolute parent_link: link_2 child_link: link_0
joint_name: joint_1 joint_type: revolute parent_link: link_2 child_link: link_1
joint_name: joint_2 joint_type: fixed parent_link: base child_link: link_2
```

```box semantics
link_0 hinge rotation_lid
link_1 hinge rotation_lid
link_2 free box_body
```

Links:
link_0: To fully open the box, the robot needs to open both box lids. We know from the semantics that link_0 is one of the lids.
link_1: To fully open the box, the robot needs to open both box lids. We know from the semantics that link_1 is another lid.
Joints:
joint_0: from the articulation tree, joint_0 connects link_0 and is a hinge joint. Thus, the robot needs to actuate joint_0 to open link_0, which is the lid of the box.
joint_1: from the articulation tree, joint_1 connects link_1 and is a hinge joint. Thus, the robot needs to actuate joint_1 to open link_1, which is the lid of the box.

This task can be decomposed as follows:

substep 1: grasp the first lid of the box
```primitive
	# The semantics shows that link_0 and link_1 are the lid links. 
	rgbs, final_state = grasp_object_link(self, "box", "link_0")  
    success = check_grasped(self, "box", "link_0")
```

substep 2: open the first lid of the box
```reward
def _compute_reward(self):
    # This reward encourages the end-effector to stay near the lid to grasp it.
    eef_pos = get_eef_pos(self)[0]
    lid_pos = get_link_state(self, "box", "link_0")
    reward_near = -np.linalg.norm(eef_pos - lid_pos)

    # Get the joint state of the first lid. The semantics and the articulation tree show that joint_0 connects link_0 and is the joint that controls the rotation of the first lid link_0.
    joint_angle = get_joint_state(self, "box", "joint_0") 
    # The reward is the negative distance between the current joint angle and the joint angle when the lid is fully open (upper limit).
    joint_limit_low, joint_limit_high = get_joint_limit(self, "box", "joint_0")
    diff = np.abs(joint_angle - joint_limit_high)
    reward_joint =  -diff

    reward = reward_near + 5 * reward_joint
    success = diff < 0.35 * (joint_limit_high - joint_limit_low)

    return reward, success
```

```action space
delta-translation
```

substep 3: grasp the second lid of the box
```primitive
	# We know from the semantics that link_0 and link_1 are the lid links. 
	rgbs, final_state = grasp_object_link(self, "box", "link_1")  
    success = check_grasped(self, "box", "link_1")
```

substep 4: open the second lid of the box
```reward
def _compute_reward(self):
    # This reward encourages the end-effector to stay near the lid to grasp it.
    eef_pos = get_eef_pos(self)[0]
    lid_pos = get_link_state(self, "box", "link_1")
    reward_near = -np.linalg.norm(eef_pos - lid_pos)

    # Get the joint state of the second lid. 
    joint_angle = get_joint_state(self, "box", "joint_1") 
    # The reward is the negative distance between the current joint angle and the joint angle when the lid is fully open (upper limit).
    joint_limit_low, joint_limit_high = get_joint_limit(self, "box", "joint_1")
    diff = np.abs(joint_angle - joint_limit_high)
    reward_joint =  -diff

    reward = reward_near + 5 * reward_joint
    success = diff < 0.35 * (joint_limit_high - joint_limit_low)
    return reward, success
```

```action space
delta-translation
```

substep 5: grasp the toy car
```primitive
	rgbs, final_state = grasp_object(self, "toy_car")
    success = check_grasped(self, "toy_car")
```

substep 6: put the toy car into the box
```reward
def _compute_reward(self):
    # This reward encourages the end-effector to stay near the car to grasp it.
    car_position = get_position(self, "toy_car")
    eef_pos = get_eef_pos(self)[0]
    reward_near = -np.linalg.norm(eef_pos - car_position)

    # main reward is 1 if the car is inside the box. From the semantics we know that link2 is the box body
    box_bbox_low, box_bbox_high = get_bounding_box_link(self, "box", "link_2")
    reward_in = int(in_bbox(self, car_position, box_bbox_low, box_bbox_high))
    
    # another reward is to encourage the robot to move the car to be near the box
    reward_reaching = - np.linalg.norm(car_position - (box_bbox_low + box_bbox_high) / 2)

    # The task is considered to be successful if the car is inside the box bounding box
    success = reward_in

    # We give more weight to reward_in, which is the major goal of the task.
    reward = 5 * reward_in + reward_reaching + reward_near
    return reward, success
```

```action space
normalized-direct-translation
```
Since this substep requires moving the item to a target location, we use the normalized-direct-translation.

Please decompose the following task into substeps. For each substep, write a primitive/a reward function, write the success checking function, and the action space if the reward is used. 

The primitives you can call:
grasp_object(self, object_name): the robot arm will grasp the object specified by the argument object name.
grasp_object_link(self, object_name, link_name): some object like an articulated object is composed of multiple links. The robot will grasp a link with link_name on the object with object_name. 
release_grasp(self): the robot will release the grasped object.
Note that all primitives will return a tuple (rgbs, final_state) which represents the rgb images of the execution process and the final state of the execution process. 
You should always call the primitive in the following format:
rgbs, final_state = some_primitive_function(self, arg1, ..., argn)

The APIs you can use for writing the reward function/success checking function:
get_position(self, object_name): get the position of center of mass of object with object_name.
get_orientation(self, object_name): get the orientation of an object with object_name.
get_joint_state(self, object_name, joint_name): get the joint angle value of a joint in an object.
get_joint_limit(self, object_name, joint_name): get the lower and upper joint angle limit of a joint in an object, returned as a 2-element tuple.
get_link_state(self, object_name, link_name): get the position of the center of mass of the link of an object.
get_eef_pos(self): returns the position, orientation of the robot end-effector as a list.
get_bounding_box(self, object_name): get the axis-aligned bounding box of an object. It returns the min and max xyz coordinate of the bounding box.
get_bounding_box_link(self, object_name, link_name): get the axis-aligned bounding box of the link of an object. It returns the min and max xyz coordinate of the bounding box.
in_bbox(self, pos, bbox_min, bbox_max): check if pos is within the bounding box with the lowest corner at bbox_min and the highest corner at bbox_max. 
check_grasped(self, object_name, link_name): return true if an object or a link of the object is grasped. link_name can be none, in which case it will check whether the object is grasped.
get_initial_pos_orient(self, obj): get the initial position and orientation of an object at the beginning of the task.
get_initial_joint_angle(self, obj_name, joint_name): get the initial joint angle of an object at the beginning of the task.

The action space you can use for learning with the reward: delta-translation is better suited for small movements, and normalized-direct-translation is better suited for directly specifying the target location of the robot end-effector. 
You can assume that for objects, the lower joint limit corresponds to their natural state, e.g., a box is closed with the lid joint being 0, and a lever is unpushed when the joint angle is 0.
"""
]

assistant_contents = [
"""
Yes, I understand the goal. Please proceed with the next example.
""",

"""
Yes, I understand the goal. Please proceed with the next example.
"""
]



reward_file_header1 = """
from manipulation.sim import SimpleEnv
import numpy as np
from manipulation.gpt_reward_api import *
from manipulation.gpt_primitive_api import *
import gym

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

primitive_file_header1 = """
from manipulation.sim import SimpleEnv
import numpy as np
from manipulation.gpt_primitive_api import *
from manipulation.gpt_reward_api import *
import gym

class {}(SimpleEnv):
"""

primitive_file_header2 = """
    def __init__(self, task_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_name = task_name
        self.detected_position = {}

    def execute(self):
"""

primitive_file_end = """
        return rgbs, final_state, success

gym.register(
    id='{}-v0',
    entry_point={},
)
"""

def decompose_and_generate_reward_or_primitive(task_name, task_description, initial_config, articulation_tree, semantics, 
                              involved_links, involved_joints, object_id, yaml_config_path, save_path, 
                              temperature=0.4, model='gpt-4'):
    query_task = """
Task name: {}
Description: {}
Initial config:
```yaml
{}
```

{}

{}

Links:
{}
Joints:
{}
""".format(task_name, task_description, initial_config, articulation_tree, semantics, involved_links, involved_joints)
    
    filled_user_contents = copy.deepcopy(user_contents)
    filled_user_contents[-1] = filled_user_contents[-1] + query_task

    system = "You are a helpful assistant."
    reward_response = query(system, filled_user_contents, assistant_contents, save_path=save_path, debug=False, 
                            temperature=temperature, model=model)
    res = reward_response.split("\n")

    substeps = []
    substep_types = []
    reward_or_primitives = []
    action_spaces = []

    num_lines = len(res)
    for l_idx, line in enumerate(res):
        line = line.lower()
        if line.startswith("substep"):
            substep_name = line.split(":")[1]
            substeps.append(substep_name)

            py_start_idx, py_end_idx = l_idx, l_idx
            for l_idx_2 in range(l_idx + 1, num_lines):
                ### this is a reward
                if res[l_idx_2].lower().startswith("```reward"):
                    substep_types.append("reward")
                    py_start_idx = l_idx_2 + 1
                    for l_idx_3 in range(l_idx_2 + 1, num_lines):
                        if "```" in res[l_idx_3]:
                            py_end_idx = l_idx_3
                            break
            
                if res[l_idx_2].lower().startswith("```primitive"):
                    substep_types.append("primitive")
                    action_spaces.append("None")
                    py_start_idx = l_idx_2 + 1
                    for l_idx_3 in range(l_idx_2 + 1, num_lines):
                        if "```" in res[l_idx_3]:
                            py_end_idx = l_idx_3
                            break
                    break

                if res[l_idx_2].lower().startswith("```action space"):
                    action_space = res[l_idx_2 + 1]
                    action_spaces.append(action_space)
                    break

            reward_or_primitive_lines = res[py_start_idx:py_end_idx]
            reward_or_primitive_lines = [line.lstrip() for line in reward_or_primitive_lines]
            if substep_types[-1] == 'reward':
                reward_or_primitive_lines[0] = "    " + reward_or_primitive_lines[0]
                for idx in range(1, len(reward_or_primitive_lines)):
                    reward_or_primitive_lines[idx] = "        " + reward_or_primitive_lines[idx]
            else:
                for idx in range(0, len(reward_or_primitive_lines)):
                    reward_or_primitive_lines[idx] = "        " + reward_or_primitive_lines[idx]
            reward_or_primitive = "\n".join(reward_or_primitive_lines) + "\n"

            reward_or_primitives.append(reward_or_primitive)

    task_name = task_name.replace(" ", "_")
    parent_folder = os.path.dirname(os.path.dirname(save_path))
    task_save_path = os.path.join(parent_folder, "task_{}".format(task_name))
    if not os.path.exists(task_save_path):
        os.makedirs(task_save_path)

    print("substep: ", substeps)
    print("substep types: ", substep_types)
    print("reward or primitives: ", reward_or_primitives)
    print("action spaces: ", action_spaces)

    with open(os.path.join(task_save_path, "substeps.txt"), "w") as f:
        f.write("\n".join(substeps))
    with open(os.path.join(task_save_path, "substep_types.txt"), "w") as f:
        f.write("\n".join(substep_types))
    with open(os.path.join(task_save_path, "action_spaces.txt"), "w") as f:
        f.write("\n".join(action_spaces))
    with open(os.path.join(task_save_path, "config_path.txt"), "w") as f:
        f.write(yaml_config_path)

    for idx, (substep, type, reward_or_primitive) in enumerate(zip(substeps, substep_types, reward_or_primitives)):
        substep = substep.lstrip().replace(" ", "_")
        substep = substep.replace("'", "")
        file_name = os.path.join(task_save_path, f"{substep}.py")

        if type == 'reward':
            header = reward_file_header1.format(substep)
            end = reward_file_end.format(substep, substep)
            file_content =  header + reward_file_header2 + reward_or_primitive + end
            with open(file_name, "w") as f:
                f.write(file_content)
        elif type == 'primitive':
            header = primitive_file_header1.format(substep)
            end = primitive_file_end.format(substep, substep)
            file_content = header + primitive_file_header2 + reward_or_primitive + end
            with open(file_name, "w") as f:
                f.write(file_content)

    return task_save_path


