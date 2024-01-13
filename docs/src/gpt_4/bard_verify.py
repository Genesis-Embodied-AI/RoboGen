import os
import requests
import objaverse
from PIL import Image
from gpt_4.query import query
import torch
import numpy as np
from lavis.models import load_model_and_preprocess
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, vis_processors, _ = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True, device=device)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def bard_verify(image):
    from bardapi import Bard
    token = "" # replace with your token
    session = requests.Session()
    session.headers = {
                "Host": "bard.google.com",
                "X-Same-Domain": "1",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
                "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
                "Origin": "https://bard.google.com",
                "Referer": "https://bard.google.com/",
            }
    session.cookies.set("__Secure-1PSID", token) 
    bard = Bard(token=token, session=session)

    
    query_string = """I will show you an image. Please describe the content of the image. """

    print("===================== querying bard: ==========================")
    print(query_string)
    res = bard.ask_about_image(query_string, image)
    description = res['content']
    print("bard description: ", description)
    print("===============")
    return description
    
def blip2_caption(image):
    # preprocess the image
    # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
    image = vis_processors["eval"](image).unsqueeze(0).to(device)
    # generate caption
    res = model.generate({"image": image})
    return res[0]

def verify_objaverse_object(object_name, uid, task_name=None, task_description=None, use_bard=False, use_blip2=True):
    annotations = objaverse.load_annotations([uid])[uid]
    thumbnail_urls = annotations['thumbnails']["images"]

    max_size = -1000
    max_url = -1
    for dict in thumbnail_urls:
        width = dict["width"]
        if width > max_size:
            max_size = width
            max_url = dict["url"]
    if max_url == -1: # TODO: in this case, we should render the object using blender to get the image.
        return False
    
    # download the image from the url
    try: 
        raw_image = Image.open(requests.get(max_url, stream=True).raw).convert('RGB')
    except:
        return False
    
    if not os.path.exists('objaverse_utils/data/images'):
        os.makedirs('objaverse_utils/data/images')
        
    raw_image.save("objaverse_utils/data/images/{}.jpeg".format(uid))
    bard_image = open("objaverse_utils/data/images/{}.jpeg".format(uid), "rb").read()

    descriptions = []
    if use_bard:
        bard_description = bard_verify(bard_image)
        descriptions.append(bard_description)
    if use_blip2:
        blip2_description = blip2_caption(raw_image)
        descriptions.append(blip2_description)

    gpt_results = []

    for description in descriptions:
        if description:
            system = "You are a helpful assistant."
            query_string = """
            A robotic arm is trying to solve a task to learn a manipulation skill in a simulator.
        We are trying to find the best objects to load into the simulator to build this task for the robot to learn the skill.
        The task the robot is trying to learn is: {}. 
        A more detailed description of the task is: {}.
        As noted, to build the task in the simulator, we need to find this object: {}.
        We are retrieving the object from an existing database, which provides some language annotations for the object.
        With the given lanugage annotation, please think if the object can be used in the simulator as {} for learning the task {}.

        This is the language annotation:
        {}

        Please reply first with your reasoning, and then a single line with "**yes**" or "**no**" to indicate whether this object can be used.
        """.format(task_name, task_description, object_name, object_name, task_name, description)
        
            if not os.path.exists('data/debug'):
                os.makedirs('data/debug')
            res = query(system, [query_string], [], save_path='data/debug/verify.json', temperature=0)
            
            responses = res.split("\n")

            useable = False
            for l_idx, line in enumerate(responses):
                if "yes" in line.lower():
                    useable = True
                    break

            gpt_results.append(useable)

    return np.alltrue(gpt_results)

if __name__ == "__main__":
    uid = "adbd797050f5429daee730a3aad04ee3"
    verify_objaverse_object("a hamburger", uid, "Heat up a hamburger in microwave", "The robot arm places a hamburger inside the microwave, closes the door, and sets the microwave timer to heat the soup for an appropriate amount of time.")
