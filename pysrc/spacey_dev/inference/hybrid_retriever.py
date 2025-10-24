import bm25s, Stemmer, numpy as np, asyncio
from transformers import AutoModel, AutoTokenizer
import time, torch
from spacey_util.add_path import model_path, data_path, data_processed_path
import faiss, numpy as np, pandas as pd, re
from spacey_util.celestial.classifier import celestial_extract

# ======================= BM25s =======================================
stemmer = Stemmer.Stemmer("english")
retriever = bm25s.BM25.load("./data/out/bm25s")

def retrieve_lexical_ids(query, top_k: int  = 50):
    query_tokens = bm25s.tokenize(query, stemmer=stemmer)
    results, scores = retriever.retrieve(query_tokens, k=top_k)
    
    return results[0], scores[0]

# ======================= SIMCSE + FAISS ===============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FAISS_PATH = "./data/out/spacenews_sentences.faiss"

model_dir = model_path() / "princeton-nlp-sup-simcse-roberta-base"

def load_simcse_model():
    start_time = time.time()
    model = AutoModel.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    end_time = time.time()
    if __debug__:
        print(f"SimCSE Model load time: {end_time - start_time:.4f} seconds")
    
    model.to(device).eval() # evaluation mode
    return model, tokenizer


index = faiss.read_index(FAISS_PATH)

@torch.no_grad()
def encode_query(model, tokenizer, q: str, max_len=128):
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


def retrieve_dense_ids(model, tokenizer, question, top_k: int  = 50):
    question_vector = encode_query(model, tokenizer, question)

    scores, ids = index.search(question_vector, top_k)

    return ids[0], -scores[0] # L2

# RANKER
def z(x):
    x = np.array(x, dtype=float)
    sd = x.std()
    if sd < 1e-12:   # all equal -> return zeros (no effect in fusion)
        return np.zeros_like(x)
    return (x - x.mean()) / (sd + 1e-12)

def fused_retriever(model, tokenizer, question, corpus_df, top_k: int = 11, w_bm25=0.7, w_faiss=0.3):
    bm25_ids, bm25_scores = retrieve_lexical_ids(question, top_k=100)
    
    faiss_ids, faiss_sims = retrieve_dense_ids(model, tokenizer, question, top_k=100)

    q_bodies, q_properties = celestial_extract(question)
    
    ids_concat = np.concatenate([bm25_ids, faiss_ids]).tolist()
    ids = list(dict.fromkeys(ids_concat))

    if len(q_bodies) > 0 or len(q_properties) > 0:
        selected_df = corpus_df.iloc[ids]

        filter_mask = selected_df.apply(
            lambda x: (
                any(t in x['bodies'] for t in q_bodies) or 
                any(t in x['property'] for t in q_properties)
            ),
            axis=1
        )

        # if present, filter on both
        if len(q_bodies) > 0 and len(q_properties) > 0:
            filter_mask = selected_df.apply(
                lambda x: (
                    any(t in x['bodies'] for t in q_bodies) and
                    any(t in x['property'] for t in q_properties)
                ),
                axis=1
            )
        
        filtered = selected_df[filter_mask]
        ids = filtered.index
    
    bmap = {i: s for i, s in zip(bm25_ids, bm25_scores)}
    fmap = {i: s for i, s in zip(faiss_ids, faiss_sims)}
    B = np.array([bmap.get(i, 0.0) for i in ids], dtype=float)
    F = np.array([fmap.get(i, 0.0) for i in ids], dtype=float)
    S = w_bm25 * z(B) + w_faiss * z(F)
    order = np.argsort(-S)[:top_k]


    return [ids[i] for i in order], S[order]