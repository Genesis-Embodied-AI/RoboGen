user_contents_v2 = [
"""
A robotic arm is trying to manipulate some objects to learn corresponding skills in a simulator. However, the size of the objects might be wrong. Your task is to adjust the size of the objects, such that they match each other when interact with each other; and the size should also match what is commonly seen in everyday life, in household scenarios. 

Now I will give you the name of the task, the object and their sizes, please correct any unreasonable sizes. 

Objects are represented using a mesh file, you can think of size as the longest dimension of the object. 

I will write in the following format:
```
Task: task description
obj1, mesh, size 
obj2, mesh, size
```

Please reply in the following format:
explanations of why some size is not reasonable.
```yaml
obj1, mesh, corrected_size
obj2, mesh, corrected_radius
```

Here is an example:
Input: 
```
Task: The robotic arm lowers the toilet seat from an up position to a down position
Toilet, mesh, 0.2
```

Output:
A toilet is usually 0.6 - 0.8m in its back height, so the size is not reasonable -- it is a bit too small. Below is the corrected size.
```yaml
Toilet, mesh, 0.7
```

Another example:
Input:
```
Task: Fill a cup with water under the faucet
Faucet, mesh, 0.25
Cup, mesh, 0.3
```

Output:
The size of the faucet makes senes. However, the size of the cup is too large for 2 reasons: it does not match the size of tha faucet for getting water under the faucet; and it is not a common size of cup in everyday life. Below is the corrected size.
```yaml
Faucet, mesh, 0.25 
Cup, mesh, 0.12 
```

One more example to show that even if no change is needed, you should still reply with the same size.
Input:
```
Task: Open Table Drawer The robotic arm will open a table drawer
table, mesh, 0.8
```

Output:
The size of the table is reasonable, so no change is needed.
```yaml
table, mesh, 0.8
```
This is also a good example to show that sometimes, the task description might include two objects, e.g., a table and a drawer, yet there is only one object size provided (here the table). This is not an error, but that the other object is part of the provided object, i.e., here the drawer is part of the table. It's fine, you should then just reply with the corrected size of the object provided, here, the table, in such cases.

Another example showing that sometimes we will ask you to adjust distractor objects needed for the task, instead of the main objects themselves. 
In such case (and in all cases), you just need to adjust the sizes of the provided objects, instead of asking why the main objects are not includes.
Input:
```
Task: Heat up a bowl of soup in the microwave
plate, mesh, 0.3
sponge, mesh, 0.1
oven, mesh, 0.4
```

Output:
The size of the sponge makse sense. However, the size of the plate is too big, and the size of the oven is too small.
```yaml
plate, mesh, 0.15
sponge, mesh, 0.1
oven, mesh, 0.8
```
As noted, here the main objects for the task, the microwave and the bowl of soup, are not included in the input. Instead, some distractor objects in the scene are provided. This is totally fine, you just need to correct the size of the provided objects.
"""
]

assistant_contents_v2 = [
"""
Sure, I'm ready. Please provide the task and object information.
"""
]

