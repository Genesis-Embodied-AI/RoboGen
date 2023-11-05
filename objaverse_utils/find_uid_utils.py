import pandas as pd
import torch
from objaverse_utils.utils import text_to_uid_dict
from gpt_4.verification import check_text_similarity
import numpy as np
from gpt_4.bard_verify import verify_objaverse_object
import json

objaverse_csv = pd.read_csv('objaverse_utils/Cap3D_automated_Objaverse.csv')
objaverse_csv = objaverse_csv.dropna()
objaverse_csv_uids = list(objaverse_csv.iloc[:, 0].values)
objaverse_csv_annotations = list(objaverse_csv.iloc[:, 1].values)
objaverse_csv_annotations_embeddings = torch.load("objaverse_utils/data/cap3d_sentence_bert_embeddings.pt")
tag_uids = []
tag_embeddings = []
tag_descriptions = [] 
num_chunks = 31
for idx in range(num_chunks):
    uids = torch.load("objaverse_utils/data/default_tag_uids_{}.pt".format(idx))
    embeddings = torch.load("objaverse_utils/data/default_tag_embeddings_{}.pt".format(idx))
    descriptions = torch.load("objaverse_utils/data/default_tag_names_{}.pt".format(idx))
    tag_uids = tag_uids + uids
    tag_descriptions = tag_descriptions + descriptions
    tag_embeddings.append(embeddings)

def find_uid(obj_descrption, candidate_num=10, debug=False, task_name=None, task_description=None):
    uids = text_to_uid_dict.get(obj_descrption, None)
    all_uid_candidates = text_to_uid_dict.get(obj_descrption + "_all", None)

    if uids is None:
        print("searching whole objaverse for: ", obj_descrption)
        similarities = []
        for idx in range(num_chunks):
            similarity = check_text_similarity(obj_descrption, None, check_embeddings=tag_embeddings[idx])
            similarity = similarity.flatten()
            similarities.append(similarity)
        
        similarity = check_text_similarity(obj_descrption, None, check_embeddings=objaverse_csv_annotations_embeddings)
        similarity = similarity.flatten()
        similarities.append(similarity)
        similarities = np.concatenate(similarities)

        all_uids = tag_uids + objaverse_csv_uids
        all_description = tag_descriptions + objaverse_csv_annotations

        sorted_idx = np.argsort(similarities)[::-1]

        usable_uids = []
        all_uid_candidates = [all_uids[sorted_idx[i]] for i in range(candidate_num)]
        for candidate_idx in range(candidate_num):
            print("{} candidate {} similarity: {} {}".format("=" * 10, candidate_idx, similarities[sorted_idx[candidate_idx]], "=" * 10))
            print("found uid: ", all_uids[sorted_idx[candidate_idx]])
            print("found description: ", all_description[sorted_idx[candidate_idx]])

            candidate_uid = all_uids[sorted_idx[candidate_idx]]
            bard_verify_result = verify_objaverse_object(obj_descrption, candidate_uid, task_name=task_name, task_description=task_description) # TODO: add support for including task name in the checking process
            print("{} Bard thinks this object is usable: {} {}".format("=" * 20, bard_verify_result, "=" * 20))
            if bard_verify_result:
                usable_uids.append(candidate_uid)

        if len(usable_uids) == 0:
            print("no usable objects found for {} skipping this task!!!".format(obj_descrption))
            usable_uids = all_uid_candidates

        text_to_uid_dict[obj_descrption] = usable_uids
        text_to_uid_dict[obj_descrption + "_all"] = all_uid_candidates
        with open("objaverse_utils/text_to_uid.json", 'w') as f:
            json.dump(text_to_uid_dict, f, indent=4)
        return usable_uids
    else:
        return uids
