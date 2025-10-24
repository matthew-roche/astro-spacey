from transformers import AutoModel, AutoTokenizer
import time, torch
from spacey_util.add_path import model_path
import faiss, numpy as np, pandas as pd, re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FAISS_PATH = "./data/out/spacenews_sentences.faiss"

model_dir = model_path() / "princeton-nlp-sup-simcse-roberta-base"

start_time = time.time()
model = AutoModel.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
end_time = time.time()
print(f"Model load time: {end_time - start_time:.4f} seconds")

model.to(device).eval() # evaluation mode
index = faiss.read_index(FAISS_PATH)

@torch.no_grad()
def encode_query(q: str, max_len=412):
    t = tokenizer(q, max_length=max_len, truncation=True, padding=True, return_tensors="pt").to(device)
    out = model(**t, output_hidden_states=True, return_dict=True)
    if out.pooler_output is not None:
        v = out.pooler_output
    else:
        last = out.last_hidden_state
        mask = t["attention_mask"].unsqueeze(-1)
        v = (last * mask).sum(1) / mask.sum(1).clamp(min=1)
    v = torch.nn.functional.normalize(v, p=2, dim=1)
    return v.cpu().numpy().astype("float32")  # [1, D]


def retrieve_dense_ids(question, top_k: int  = 10, index_k: int = 10):
    question_vector = encode_query(question)
    scores, ids = index.search(question_vector, index_k)

    print(index.metric_type == faiss.METRIC_L2)
    print(index.metric_type == faiss.METRIC_INNER_PRODUCT)

    return ids[0][:top_k], scores[0][:top_k]