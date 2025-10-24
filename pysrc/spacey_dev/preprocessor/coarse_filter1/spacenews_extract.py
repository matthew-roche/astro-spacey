from transformers import AutoModel, AutoTokenizer
import time, torch, json
from spacey_util.add_path import model_path, data_path, data_processed_path
import faiss, numpy as np, pandas as pd, re
from spacey_dev.preprocessor.rules.planet_rules import passes_entity_filter, boost_score, solar_system_gate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mode_dir = model_path() / "princeton-nlp-sup-simcse-roberta-large"

start_time = time.time()
model = AutoModel.from_pretrained(mode_dir)
tokenizer = AutoTokenizer.from_pretrained(mode_dir)
end_time = time.time()
print(f"Model load time: {end_time - start_time:.4f} seconds")

model.to(device).eval() # evaluation mode

df = pd.read_parquet(data_processed_path() / "spacenews.parquet")  # must align with embedding order used to build index
index = faiss.read_index("./data/processed/spacenews_index.faiss")

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

def cosine_from_l2sq(d2):  # if index is L2 and vectors are L2-normalized
    return 1.0 - (d2 / 2.0)

SEARCH_K = 100
SELECT_K = 10
COSINE_SIM_THRESHOLD = 0.5
rows = []
with open("./query/planet_query.json", "r", encoding="utf-8") as f:
    queries = json.load(f)
    all_queries = [q for buckets in queries.values() for qlist in buckets.values() for q in qlist]
    #print(all_queries[:10])

    rows = []
    for planet, categories in queries.items():
        planet_list = []
        print(f"Processing query for {planet}")
        for category, qlist in categories.items():
            for query in qlist:
                print(f"query: {query}")
                query_vector = encode_query(query)
                scores, ids = index.search(query_vector, SEARCH_K)

                # sims = cosine_from_l2sq(scores[0]) if index.metric_type == faiss.METRIC_L2 else scores[0]
                if index.metric_type == faiss.METRIC_L2:
                    sims = 1.0 - scores[0] / 2.0     # convert L2^2 to cosine
                elif index.metric_type == faiss.METRIC_INNER_PRODUCT:
                    sims = scores[0]                 # already cos if normalized
                else:
                    raise ValueError("Unknown metric")

                order = np.argsort(-sims) # reverse by desc order of similarity

                for j in order[:SELECT_K]:
                    # print(j)
                    # print(order)
                    # print(ids)
                    doc_id = int(ids[0][j])
                    sim = float(sims[j])
                    if doc_id < 0 or sim < COSINE_SIM_THRESHOLD: 
                        continue

                    title   = df.loc[doc_id, "title"]
                    content = df.loc[doc_id, "content"] if "content" in df.columns else ""
                    # abstract = df.loc[doc_id, "postexcerpt"] if "postexcerpt" in df.columns else ""

                    if not passes_entity_filter(planet, title, content, require_alias=False):
                        continue

                    if not solar_system_gate(planet, title, content):
                        continue

                    sim_adj = sim + boost_score(planet, title, content)

                    planet_list.append({
                        "id": doc_id, 
                        "similarity": sim,
                        "adjusted_sim": sim_adj,
                        "title": title,
                        "content": content,
                        # "postexcerpt": abstract # not needed
                    })
        if (len(planet_list)):
            planet_df = pd.DataFrame(planet_list)
            planet_df = planet_df.sort_values("adjusted_sim", ascending=False)
            planet_df = planet_df.drop_duplicates(subset="id", keep="first").reset_index(drop=True)
            # print(planet_df)
            planet_df.to_csv(data_processed_path() / f"spacenews_planet_{planet}.csv", index=False)
        else:
            print("Nothing found for ", planet)

        rows.extend(planet_list)
        

new_df = pd.DataFrame(rows)
new_df = new_df.sort_values("adjusted_sim", ascending=False)
new_df = new_df.drop_duplicates(subset="id", keep="first").reset_index(drop=True)

print(new_df)

new_df.to_parquet(data_processed_path() / "spacenews_extracted.parquet", index=False)