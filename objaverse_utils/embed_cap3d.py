import objaverse
import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch

csv = pd.read_csv('objaverse_utils/Cap3D_automated_Objaverse.csv')
csv = csv.dropna()
uids = csv.iloc[:, 0].values
annotations = csv.iloc[:, 1].values

sentence_bert_model = SentenceTransformer('all-mpnet-base-v2')

all_list = list(annotations)
with torch.no_grad():
    all_embeddings = emb_to_check = sentence_bert_model.encode(all_list)
torch.save(all_embeddings, "objaverse_utils/data/cap3d_sentence_bert_embeddings.pt")