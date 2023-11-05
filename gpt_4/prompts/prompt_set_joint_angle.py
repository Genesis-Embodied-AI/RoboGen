from gpt_4.query import query
import copy

user_contents = [
"""
Your goal is to set the  joint angles of some articulated objects to the right value in the initial state, given a task. The task is for a robot arm to learn the corresponding skills to manipulate the articulated object. 

The input to you will include the task name, a short description of the task, the articulation tree of the articulated object, a semantic file of the articulated object, the links and joints of the articulated objects that will be involved in the task, and the substeps for doing the task. 

You should output for each joint involved in the task, what joint value it should be set to. You should output a number in the range [0, 1], where 0 corresponds to the lower limit of that joint angle, and 1 corresponds to the upper limit of the joint angle. You can also output a string of "random", which indicates to sample the joint angle within the range.

By default, the joints in an object are set to their lower joint limits. You can assume that the lower joint limit corresponds to the natural state of the articulated object. E.g., for a door's hinge joint, 0 means it is closed, and 1 means it is open. For a lever, 0 means it is unpushed, and 1 means it is pushed to the limit. 

Here are two examples:

Input:
Task Name: Close the door
Description: The robot arm will close the door after it was opened. 


```door articulation tree
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

```door semantics
link_0 hinge rotation_door
link_1 static door_frame
link_2 hinge rotation_door
```

Links: 
- link_0: link_0 is the door. This is the part of the door assembly that the robot needs to interact with.
Joints:
- joint_0: Joint_0 is the revolute joint connecting link_0 (the door) as per the articulation tree. The robot needs to actuate this joint cautiously to ensure the door is closed.

substeps:
approach the door	
close the door


Output:
The goal is for the robot arm to learn to close the door after it is opened. Therefore, the door needs to be initially opened, thus, we are setting its value to 1, which corresponds to the upper joint limit. 
```joint values
joint_0: 1
```

Another example:
Task Name: Turn Off Faucet
Description: The robotic arm will turn the faucet off by manipulating the switch

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
- link_0: link_0 is the door. This is the part of the door assembly that the robot needs to interact with.
Joints:
- joint_0: Joint_0 is the revolute joint connecting link_0 (the door) as per the articulation tree. The robot needs to actuate this joint cautiously to ensure the door is closed.

substeps:
grasp the faucet switch
turn off the faucet

Output:
For the robot to learn to turn off the faucet, it cannot be already off initially. Therefore, joint_1 should be set to its upper joint limit, or any value that is more than half of the joint range, e.g., 0.8.
```joint value
joint_1: 0.8
```

One more example:
Task Name: Store an item inside the Drawer
Description: The robot arm picks up an item and places it inside the drawer of the storage furniture

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
- link_2: link_2 is the drawer link from the semantics. The robot needs to open this drawer to place the item inside. 
Joints: 
- joint_2: joint_2, from the articulation tree, connects to link_2 (the drawer). Thus, the robot would need to actuate this joint to open the drawer to store the item.

substeps:
 grasp the drawer
 open the drawer
 grasp the item
 put the item into the drawer
 grasp the drawer again
 close the drawer
 release the grasp


Output:
This task involves putting one item into the drawer of the storage furniture. As noted in the substeps, the robot needs to first open the drawer, put the item in, and then close it. Since the articulated object is initialized with the lower joint limit, i.e., the drawer is initially closed, it aligns with the task where the robot needs to first learn to open the drawer. Therefore, no particular joint angle needs to be set, and we just output None. 

```joint value
None
```

One more example:
Task Name: Direct Lamp light
Description: The robot positions both the head and rotation bar to direct the light at a specific object or area


```Lamp articulation tree
links: 
base
link_0
link_1
link_2
link_3

joints: 
joint_name: joint_0 joint_type: revolute parent_link: link_3 child_link: link_0
joint_name: joint_1 joint_type: revolute parent_link: link_0 child_link: link_1
joint_name: joint_2 joint_type: fixed parent_link: base child_link: link_2
joint_name: joint_3 joint_type: revolute parent_link: link_2 child_link: link_3

```


```Lamp semantics
link_0 hinge rotation_bar
link_1 hinge head
link_2 free lamp_base
link_3 hinge rotation_bar

```

Links:
link_0 and link_1: These two links are necessary to direct the lamp light toward a specific area because they represent the rotation bar and lamp head respectively.

Joints:
joint_0 and joint_1: These joints connect the rotation bar and the lamp head. By actuating both these joints, the robot can direct the light at a desired location.

substeps:
 grasp the first rotation bar
 rotate the first rotation bar to aim the lamp
 release the first rotation bar
 grasp the lamp head
 rotate the lamp head to aim the lamp
 release the lamp head

Output:
The task involves directing the lamp light at a specific area. The robot needs to learn to manipulate both the rotation bar and the lamp head to achieve this. Therefore, we need to set the initial joint angles such that the lamp is not already directed at the desired area. We can set both joint_0 and joint_1 to be randomly sampled.

```joint values
joint_0: random
joint_1: random
```

Can you do it for the following task:
"""
]

assistant_contents = []

def query_joint_angle(task_name, task_description, articulation_tree, semantics, links, joints, substeps, save_path=None, 
                      temperature=0.1, model='gpt-4'):
    input = """
Task Name: {}
Description: {}

{}

{}

Links:
{}

Joints:
{}

substeps:
{}
""".format(task_name, task_description, articulation_tree, semantics, links, joints, "".join(substeps))
    
    new_user_contents = copy.deepcopy(user_contents)
    new_user_contents[0] = new_user_contents[0] + input

    if save_path is None:
        save_path = 'data/debug/{}_joint_angle.json'.format(input_task_name.replace(" ", "_"))

    system = "You are a helpful assistant."
    response = query(system, new_user_contents, assistant_contents, save_path=save_path, temperature=temperature, model=model)

    # TODO: parse the response to get the joint angles
    response = response.split("\n")

    joint_values = {}
    for l_idx, line in enumerate(response):
        if line.lower().startswith("```joint values"):
            for l_idx_2 in range(l_idx+1, len(response)):
                if response[l_idx_2].lower().startswith("```"):
                    break
                if response[l_idx_2].lower().strip() == "none":
                    continue
                joint_name, joint_value = response[l_idx_2].split(":")
                joint_values[joint_name.strip().lstrip()] = joint_value.strip().lstrip()

    return joint_values
