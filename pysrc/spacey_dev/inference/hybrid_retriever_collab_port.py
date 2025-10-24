# COLLAB PORT v0.1.0
import bm25s, Stemmer, numpy as np, json, faiss, time, torch, re
from transformers import AutoModel, AutoTokenizer

# ======================= BM25s =======================================
stemmer = Stemmer.Stemmer("english")
retriever = bm25s.BM25.load("./spaceyRetriever/index/bm25s")

def retrieve_lexical_ids(query, top_k: int  = 50):
    query_tokens = bm25s.tokenize(query, stemmer=stemmer)
    results, scores = retriever.retrieve(query_tokens, k=top_k)
    
    return results[0], scores[0]

# ======================= SIMCSE + FAISS ===============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FAISS_PATH = "./spaceyRetriever/index/spacenews_sentences.faiss"
simcse_model = "princeton-nlp/unsup-simcse-roberta-large"

start_time = time.time()
model = AutoModel.from_pretrained(simcse_model)
tokenizer = AutoTokenizer.from_pretrained(simcse_model)
end_time = time.time()
if __debug__:
    print(f"Model load time: {end_time - start_time:.4f} seconds")

model.to(device).eval() # evaluation mode
index = faiss.read_index(FAISS_PATH)

@torch.no_grad()
def encode_query(q: str, max_len=128):
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

# get top 50
def retrieve_dense_ids(question, top_k: int  = 50):
    question_vector = encode_query(question)
    scores, ids = index.search(question_vector, top_k)

    return ids[0], -scores[0] # L2 conversion

# z-score normalize
def z(x):
    x = np.array(x, dtype=float)
    sd = x.std()
    if sd < 1e-12:
        return np.zeros_like(x)
    return (x - x.mean()) / (sd + 1e-12)


def hybrid_retrieve(question, top_k: int = 10, w_bm25=0.7, w_faiss=0.3):
    """Return a list of ids on indexed corpus (spacenews_sentences)

    Parameters
    ----------
    question : str
        query text
    top_k : int, optional
        number of ids to return
    w_bm25: int, optional
        bm25 weight during retrieval
    w_faiss: int, optional
        faiss weight during retrieval
    
    Returns
    ----------
    ids: list(int)
        returns a list of indexed ids from highest scores
    scores: list(int)
        returns the list of scores for the ids
    """
    bm25_ids, bm25_scores = retrieve_lexical_ids(question, top_k=40)
    faiss_ids, faiss_sims = retrieve_dense_ids(question, top_k=40)

    ids_concat = np.concatenate([bm25_ids, faiss_ids]).tolist()
    ids = list(dict.fromkeys(ids_concat))

    bmap = {i: s for i, s in zip(bm25_ids, bm25_scores)}
    fmap = {i: s for i, s in zip(faiss_ids, faiss_sims)}
    B = np.array([bmap.get(i, 0.0) for i in ids], dtype=float)
    F = np.array([fmap.get(i, 0.0) for i in ids], dtype=float)
    S = w_bm25 * z(B) + w_faiss * z(F)
    order = np.argsort(-S)[:top_k+1]

    return [ids[i] for i in order], S[order]

# Bundle Extractor
with open("./spaceyRetriever/planetary_vocab.json", "r") as f:
    planetary_vocab = json.load(f)

def celestial_extract(sentence:str):
    words = sentence.lower().split()

    planets_in_text = [p for p in planetary_vocab["planets"] if p.lower() in words]
    moons = [p for p in planetary_vocab["moons"] if p.lower() in words]
    asteroids = [p for p in planetary_vocab["asteroids"] if p.lower() in words]
    dwplanets_in_text = [p for p in planetary_vocab["dwarf_planets"] if p.lower() in words]
    prop_in_text = [p for p in planetary_vocab["planetary_properties"] if p.lower() in words]

    observations = []
    for phrase in planetary_vocab["observations"]:
        pattern = r"\b" + re.escape(phrase.lower()) + r"\b"
        if re.search(pattern, sentence.lower()):
            observations.append(phrase)

    body = set(planets_in_text + dwplanets_in_text + moons + asteroids)
    prop = set(prop_in_text + observations)
    
    return list(body), list(prop)

def aggressive_hybrid_retriever(question, corpus_df, top_k: int = 11, w_bm25=0.7, w_faiss=0.3):
    bm25_ids, bm25_scores = retrieve_lexical_ids(question, top_k=100)
    faiss_ids, faiss_sims = retrieve_dense_ids(question, top_k=100)

    q_bodies, q_properties = celestial_extract(question)

    ids_concat = np.concatenate([bm25_ids, faiss_ids]).tolist()
    ids = list(dict.fromkeys(ids_concat))
    
    if len(q_bodies) > 0 or len(q_properties) > 0:
        selected_df = corpus_df.iloc[ids]

        filtered = selected_df[corpus_df.apply(
            lambda x: (
                any(t in x['bodies'] for t in q_bodies) or 
                any(t in x['property'] for t in q_properties)
            ),
            axis=1
        )]

        # if present, filter on both
        if len(q_bodies) > 0 and len(q_properties) > 0:
            filtered = selected_df[corpus_df.apply(
                lambda x: (
                    any(t in x['bodies'] for t in q_bodies) and
                    any(t in x['property'] for t in q_properties)
                ),
                axis=1
            )]

        ids = filtered.index

    # if question isn't related to celestial bodies, use default
    # else:
    #     ids_concat = np.concatenate([bm25_ids, faiss_ids]).tolist()
    #     ids = list(dict.fromkeys(ids_concat))

    bmap = {i: s for i, s in zip(bm25_ids, bm25_scores)}
    fmap = {i: s for i, s in zip(faiss_ids, faiss_sims)}
    B = np.array([bmap.get(i, 0.0) for i in ids], dtype=float)
    F = np.array([fmap.get(i, 0.0) for i in ids], dtype=float)
    S = w_bm25 * z(B) + w_faiss * z(F)
    order = np.argsort(-S)[:top_k]

    return [ids[i] for i in order], S[order]