import json
import os.path as osp
from collections import defaultdict

with open("data/partnet_mobility_dict.json", 'r') as f:
    partnet_mobility_dict = json.load(f)

if osp.exists("objaverse_utils/text_to_uid.json"):
    with open("objaverse_utils/text_to_uid.json", 'r') as f:
        text_to_uid_dict = json.load(f)
else:
    text_to_uid_dict = {}

if osp.exists("data/sapien_cannot_vhacd_part.json"):
    with open("data/sapien_cannot_vhacd_part.json", 'r') as f:
        sapaien_cannot_vhacd_part_dict = json.load(f)
else:
    sapaien_cannot_vhacd_part_dict = defaultdict(list)


if __name__ == '__main__':
    print(partnet_mobility_dict["Oven"])