import objaverse
from sentence_transformers import SentenceTransformer, util
import torch
import pickle5 as pickle
import pandas as pd
sentence_bert_model = SentenceTransformer('all-mpnet-base-v2')

uids = objaverse.load_uids()
annotations = objaverse.load_annotations(uids)
matched_uids = []
matched_names = []
for uid in uids:
    if "name" in annotations[uid] and annotations[uid]['name'] != "":
        matched_uids.append(uid)
        matched_names.append(annotations[uid]['name'])

    if "description" in annotations[uid] and annotations[uid]['description'] != "":
        matched_uids.append(uid)
        matched_names.append(annotations[uid]['description'])

    for tag in annotations[uid]['tags']:
        matched_uids.append(uid)
        matched_names.append(tag['name'])

all_list = matched_names
print("all_list length: ", len(all_list))

batch_size = 30
per_batch_size = len(all_list) // batch_size
for idx in range(batch_size + 1):
    print("encoding batch: ", idx)
    to_encode_names = all_list[idx * per_batch_size: (idx + 1) * per_batch_size]
    to_encode_uids = matched_uids[idx * per_batch_size: (idx + 1) * per_batch_size]
    with torch.no_grad():
        all_embeddings = sentence_bert_model.encode(to_encode_names)
        
    torch.save(all_embeddings, "objaverse_utils/data/default_tag_embeddings_{}.pt".format(idx))
    torch.save(to_encode_names, "objaverse_utils/data/default_tag_names_{}.pt".format(idx))
    torch.save(to_encode_uids, "objaverse_utils/data/default_tag_uids_{}.pt".format(idx))