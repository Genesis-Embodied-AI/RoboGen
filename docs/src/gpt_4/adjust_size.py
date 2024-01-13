from gpt_4.query import query
from gpt_4.prompts.prompt_with_scale import user_contents_v2 as scale_user_contents_v2, assistant_contents_v2 as scale_assistant_contents_v2
import yaml
import re
from manipulation.utils import parse_center
import copy

def adjust_size_v2(task_description, yaml_string, save_path, temperature=0.2, model='gpt-4'):
    # extract object names and sizes
    object_names = []
    object_sizes = []
    object_types = []

    config = yaml.safe_load(yaml_string)
    for obj in config:
        if "name" in obj:
            object_names.append(obj['name'].lower())
            object_types.append(obj['type'])
            if obj['type'] == 'mesh' or obj['type'] == 'urdf' or obj['type'] == 'sphere':
                object_sizes.append(obj['size'])
            if obj['type'] in ['cylinder', 'cube', 'box']:
                if isinstance(obj['size'], list):
                    object_sizes.append([str(x) for x in obj["size"]])
                else:
                    object_sizes.append([str(x) for x in parse_center(obj['size'])])
    
    new_user_contents = "```\n"
    better_task_description = re.sub(r'\d', '', task_description)
    better_task_description = better_task_description.replace("_", " ")
    better_task_description = better_task_description.lstrip()
    better_task_description = better_task_description.strip()
    new_user_contents += "Task: {}\n".format(better_task_description)
    for name, type, size in zip(object_names, object_types, object_sizes):
        if type in ['mesh', 'urdf', 'sphere']:
            new_user_contents += "{}, {}, {}\n".format(name, type, size)
        else:
            new_content = "{}, {}, ".format(name, type)
            size_string = ", ".join(size)
            new_content = new_content + size_string + "\n"
            new_user_contents += new_content
    new_user_contents += "```"
    input_user = copy.deepcopy(scale_user_contents_v2)
    input_user.append(new_user_contents)

    system = "You are a helpful assistant."
    response = query(system, input_user, scale_assistant_contents_v2, save_path=save_path, debug=False, temperature=temperature, model=model)

    response = response.split('\n')

    corrected_names = []
    corrected_sizes = []
    for idx, line in enumerate(response):
        if "```yaml" in line:
            for idx2 in range(idx+1, len(response)):
                line2 = response[idx2]
                if "```" in line2:
                    break
                line2 = line2.split(", ")
                corrected_names.append(line2[0].lower())
                sizes = line2[2:]
                if len(sizes) > 1:
                    corrected_sizes.append([float(x) for x in sizes])
                else:
                    corrected_sizes.append(float(sizes[0]))
    
    # replace the size in yaml
    for obj in config:
        if 'type' in obj:
            if obj['type'] == 'mesh' or obj['type'] == 'urdf':
                obj['size'] = corrected_sizes[corrected_names.index(obj['name'].lower())]

    return config