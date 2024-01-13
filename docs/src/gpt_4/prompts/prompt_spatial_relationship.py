from gpt_4.query import query
import copy

user_contents = [
"""
Your goal is to output any special spatial relationships certain objects should have in the initial state, given a task. The task is for a robot arm to learn the corresponding skills in household scenarios.  

The input to you will include 
the task name, 
a short description of the task, 
objects involved in the task, 
substeps for performing the task,
If there is an articulated object involved in the task, the articulation tree of the articulated object, the semantic file of the articulated object, and the links and joints of the articulated objects that will be involved in the task. 

We have the following spatial relationships:
on, obj_A, obj_B: object A is on top of object B, e.g., a fork on the table.
in, obj_A, obj_B: object A is inside object B, e.g., a gold ring in the safe.
in, obj_A, obj_B, link_name: object A is inside the link with link_name of object B. For example, a table might have two drawers, represented with link_0, and link_1, and in(pen, table, link_0) would be that a pen is inside one of the drawers that corresponds to link_0. 

Given the input to you, you should output any needed spatial relationships of the involved objects. 

Here are some examples:

Input:
Task Name:Fetch Item from Refrigerator 
Description: The robotic arm will open a refrigerator door and reach inside to grab an item and then close the door.
Objects involved: refrigerator, item

```refrigerator articulation tree
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

```refrigerator semantics
link_0 heavy refrigerator_body
link_1 hinge door
link_2 hinge door
```

Links:
link_1: The robot needs to approach and open this link, which represents one of the refrigerator doors, to reach for the item inside.
Joints:
joint_1: This joint connects link_1, representing one of the doors. The robot needs to actuate this joint to open the door, reach for the item, and close the door. 


substeps:
 grasp the refrigerator door
 open the refrigerator door
 grasp the item
 move the item out of the refrigerator
 grasp the refrigerator door again
 close the refrigerator door


Output:
The goal is for the robot arm to learn to retrieve an item from the refrigerator. Therefore, the item needs to be initially inside the refrigerator. From the refrigerator semantics we know that link_0 is the body of the refrigerator, therefore we should have a spatial relationship as the following:
```spatial relationship
In, item, refrigerator, link_0
```

Another example:
Task Name: Turn Off Faucet
Description: The robotic arm will turn the faucet off by manipulating the switch
Objects involved: faucet

```Faucet articulation tree
links: 
base
link_0
link_1

joints: 
joint_name: joint_0 joint_type: fixed parent_link: base child_link: link_0
joint_name: joint_1 joint_type: revolute parent_link: link_0 child_link: link_1
```

```Faucet semantics
link_0 static faucet_base
link_1 hinge switch
```

Links: 
link_0: link_0 is the door. This is the part of the door assembly that the robot needs to interact with.
Joints:
joint_0: Joint_0 is the revolute joint connecting link_0 (the door) as per the articulation tree. The robot needs to actuate this joint cautiously to ensure the door is closed.

substeps:
grasp the faucet switch
turn off the faucet

Output:
There is only 1 object involved in the task, thus no special spatial relationships are required.
```spatial relationship
None
```

One more example:
Task Name: Store an item inside the Drawer
Description: The robot arm picks up an item and places it inside the drawer of the storage furniture.
Objects involved: storage furniture, item

```StorageFurniture articulation tree
links: 
base
link_0
link_1
link_2

joints: 
joint_name: joint_0 joint_type: revolute parent_link: link_1 child_link: link_0
joint_name: joint_1 joint_type: fixed parent_link: base child_link: link_1
joint_name: joint_2 joint_type: prismatic parent_link: link_1 child_link: link_2
```

```StorageFurniture semantics
link_0 hinge rotation_door
link_1 heavy furniture_body
link_2 slider drawer
```

Links:
link_2: link_2 is the drawer link from the semantics. The robot needs to open this drawer to place the item inside. 
Joints: 
joint_2: joint_2, from the articulation tree, connects to link_2 (the drawer). Thus, the robot would need to actuate this joint to open the drawer to store the item.

substeps:
 grasp the drawer
 open the drawer
 grasp the item
 put the item into the drawer
 grasp the drawer again
 close the drawer
 release the grasp


Output:
This task involves putting one item into the drawer of the storage furniture. The item should initially be outside of the drawer, such that the robot can learn to put it into the drawer. Therefore, no special relationships of in or on are needed. Therefore, no special spatial relationships are needed.
```spatial relationship
None
```

Can you do it for the following task: 
"""
]

user_contents_rigid = [
"""
Your goal is to output any special spatial relationships certain objects should have in the initial state, given a task. The task is for a robot arm to learn the corresponding skills in household scenarios.  

The input to you will include 
the task name, 
objects involved in the task, 
substeps for performing the task.

We have the following spatial relationships:
on, obj_A, obj_B: object A is on top of object B, e.g., a fork on the table.
in, obj_A, obj_B: object A is inside object B, e.g., a gold ring in the safe.

Given the input to you, you should output any needed spatial relationships of the involved objects. 

Here are some examples:

Input:
Task Name:Fetch Item from Refrigerator 
Objects involved: refrigerator, item

substeps:
 grasp the refrigerator door
 open the refrigerator door
 grasp the item
 move the item out of the refrigerator
 grasp the refrigerator door again
 close the refrigerator door


Output:
The goal is for the robot arm to learn to retrieve an item from the refrigerator. Therefore, the item needs to be initially inside the refrigerator. Therefore we should have a spatial relationship as the following:
```spatial relationship
In, item, refrigerator
```

Another example:
Task Name: Turn Off Faucet
Objects involved: faucet

substeps:
grasp the faucet switch
turn off the faucet

Output:
There is only 1 object involved in the task, thus no special spatial relationships are required.
```spatial relationship
None
```

One more example:
Task Name: Store an item inside the Drawer
Objects involved: storage furniture, item

substeps:
 grasp the drawer
 open the drawer
 grasp the item
 put the item into the drawer
 grasp the drawer again
 close the drawer
 release the grasp


Output:
This task involves putting one item into the drawer of the storage furniture. The item should initially be outside of the drawer, such that the robot can learn to put it into the drawer. Therefore, no special relationships of in or on are needed.
```spatial relationship
None
```

Can you do it for the following task: 
"""
]

assistant_contents = []

def query_spatial_relationship(task_name, task_description, involved_objects, articulation_tree, semantics, links, joints, substeps, save_path=None, 
                               temperature=0.1, model='gpt-4'):
    input = """
Task Name: {}
Description: {}
Objects involved: {}

{}

{}

Links:
{}

Joints:
{}

substeps:
{}
""".format(task_name, task_description, involved_objects, articulation_tree, semantics, links, joints, "".join(substeps))
    
    new_user_contents = copy.deepcopy(user_contents)
    new_user_contents[0] = new_user_contents[0] + input

    if save_path is None:
        save_path = 'data/debug/{}_joint_angle.json'.format(input_task_name.replace(" ", "_"))

    system = "You are a helpful assistant."
    response = query(system, new_user_contents, assistant_contents, save_path=save_path, temperature=temperature, model=model)

    # TODO: parse the response to get the joint angles
    response = response.split("\n")

    spatial_relationships = []
    for l_idx, line in enumerate(response):
        if line.lower().startswith("```spatial relationship"):
            for l_idx_2 in range(l_idx+1, len(response)):
                if response[l_idx_2].lower().startswith("```"):
                    break
                if response[l_idx_2].lower().strip() == "none":
                    continue
                spatial_relationships.append(response[l_idx_2].strip().lstrip().lower())

    return spatial_relationships

def query_spatial_relationship_rigid(task_name, involved_objects, substeps, save_path=None, temperature=0.1, model='gpt-4'):
    input = """
Task Name: {}
Objects involved: {}

substeps:
{}
""".format(task_name, involved_objects, "".join(substeps))
    
    new_user_contents = copy.deepcopy(user_contents_rigid)
    new_user_contents[0] = new_user_contents[0] + input

    if save_path is None:
        save_path = 'data/debug/{}_joint_angle.json'.format(input_task_name.replace(" ", "_"))

    system = "You are a helpful assistant."
    response = query(system, new_user_contents, assistant_contents, save_path=save_path, temperature=temperature, model=model)

    # TODO: parse the response to get the joint angles
    response = response.split("\n")

    spatial_relationships = []
    for l_idx, line in enumerate(response):
        if line.lower().startswith("```spatial relationship"):
            for l_idx_2 in range(l_idx+1, len(response)):
                if response[l_idx_2].lower().startswith("```"):
                    break
                if response[l_idx_2].lower().strip() == "none":
                    continue
                spatial_relationships.append(response[l_idx_2].strip().lstrip().lower())

    return spatial_relationships
