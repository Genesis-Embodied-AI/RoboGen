import json
import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch

with open("data/partnet_mobility_dict.json", 'r') as f:
    partnet_mobility_dict = json.load(f)
obj_categories = list(partnet_mobility_dict.keys())

sentence_bert_model = SentenceTransformer('all-mpnet-base-v2')

with torch.no_grad():
    all_embeddings = emb_to_check = sentence_bert_model.encode(obj_categories)
torch.save(all_embeddings, "objaverse_utils/data/partnet_mobility_category_embeddings.pt")
