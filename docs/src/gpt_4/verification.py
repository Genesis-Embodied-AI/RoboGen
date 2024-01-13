import torch
from sentence_transformers import SentenceTransformer, util
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
sentence_bert_model = None

def check_text_similarity(text, check_list=None, check_embeddings=None):
    global sentence_bert_model
    if sentence_bert_model is None:
        sentence_bert_model = SentenceTransformer('all-mpnet-base-v2')

    #Sentences are encoded by calling model.encode()
    with torch.no_grad():
        emb1 = sentence_bert_model.encode(text)
        if check_embeddings is None:
            emb_to_check = sentence_bert_model.encode(check_list)
        else:
            emb_to_check = check_embeddings
        cos_sim = util.cos_sim(emb1, emb_to_check)

    return cos_sim.cpu().numpy()