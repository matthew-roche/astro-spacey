import numpy as np, pandas as pd, umap, hdbscan, time, re
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from spacey_util.add_path import data_path, data_processed_path

CLUSTER_LABEL_K = 1
CLUSTER_MIN_SIZE = 15
PRUNE_WORDS_MIN_DF = 0.005

DATASET_FILE = data_processed_path() / "spacenews.parquet"
SIMCSE_EMBEDDING_FILE = data_processed_path() /  "spacenews_simcse.npy"

SAVE_CLUSTERED_FILE = data_processed_path() / "spacenews_clustered.csv"
SAVE_CLUSTERED_LABELS = data_processed_path() / "spacenews_clustered_labels.npy"

df  = pd.read_parquet(DATASET_FILE).fillna("")
# texts = (df["title"].fillna("") + " — " + df.get("postexcerpt","")).astype(str)

# refactor
# titles = df["title"].fillna("")
# excerpts = df.get("postexcerpt", pd.Series([""]*len(df))).fillna("")
# contents = df.get("content", pd.Series([""]*len(df))).fillna("")
titles = df["title"]
excerpts = df["postexcerpt"]
contents = df["content"]

MIN_TOKENS = 8
def build_text(title, abstract, context):
    # prefer excerpt if it’s substantive; else use head of content; else title
    if len(abstract.split()) >= MIN_TOKENS:
        return f"{title} — {abstract}", False
    elif len(context.split()) >= MIN_TOKENS:
        # take head; we’ll stride if needed
        return f"{title} — {context}", True
    elif len(title.split()) >= 1:
        return f"{title}", False
    else:
        return "", False

pairs = [build_text(t, e, c) for t, e, c in zip(titles, excerpts, contents)]
texts_list  = [p[0] for p in pairs]
texts  = np.asarray(texts_list, dtype=object)

# umap, hdbscan
X = np.load(SIMCSE_EMBEDDING_FILE)
ump = umap.UMAP(n_neighbors=60, min_dist=0.05, n_components=50,
               metric="cosine", random_state=42).fit_transform(X)

cl = hdbscan.HDBSCAN(min_cluster_size=CLUSTER_MIN_SIZE, min_samples=5,
                     cluster_selection_method="leaf",
                     cluster_selection_epsilon=0.07,
                     metric="euclidean")

labels = cl.fit_predict(ump)
df["cluster_id"] = labels

# Cluster words
BODIES = [
    "mercury","venus","earth","mars","jupiter","saturn","uranus","neptune","pluto",
    "sun","solar","moon","europa","enceladus","titan","ganymede","callisto","triton","io","ceres"
]
PLANET_PATTERNS = {p: re.compile(rf"\b{re.escape(p)}\b", re.I) for p in BODIES}


def is_planet_word(token: str) -> bool:
    return any(rx.search(token) for rx in PLANET_PATTERNS.values())

def find_bodies(text):
    hits = []
    for b, rx in PLANET_PATTERNS.items():
        if rx.search(text):
            hits.append(b)
    return hits

def dominant_body(docs):
    c = Counter()
    for d in docs:
        c.update(find_bodies(d))
    return c.most_common(1)[0][0] if c else None

def is_redundant(term, selected):
    """Drop if term equals/contains/contained-by a selected term (case-insensitive)."""
    tl = term.lower()
    for s in selected:
        sl = s.lower()
        if tl == sl or tl.startswith(sl+" ") or tl.endswith(" "+sl) or sl in tl.split():
            return True
        if sl.startswith(tl+" ") or sl.endswith(" "+tl) or tl in sl.split():
            return True
    return False

def topic_words_distinct(docs, top_k=10, min_df=2, max_df=0.9):
    # TF-IDF over 1–2 grams
    cv = CountVectorizer(
        stop_words="english",
        ngram_range=(1,2),
        min_df=min_df,
        max_df=max_df,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-]+\b"
    )
    Xc = cv.fit_transform(docs)
    tfidf = TfidfTransformer(norm=None, use_idf=True).fit_transform(Xc)
    vocab = np.array(cv.get_feature_names_out())
    scores = np.asarray(tfidf.sum(axis=0)).ravel()
    order = np.argsort(-scores)
    words = vocab[order].tolist()

    dom = dominant_body(docs)  # choose one body per cluster label set

    # 1) prioritize phrases that include the dominant body (prefer bigrams over the raw body)
    prioritized = []
    if dom:
        dom_regex = re.compile(rf"\b{re.escape(dom)}\b", re.I)
        # first pass: bigrams containing the body
        for w in words:
            if " " in w and dom_regex.search(w):
                prioritized.append(w)
        # if none, fall back to the bare body token
        if not prioritized:
            for w in words:
                if dom_regex.fullmatch(w):
                    prioritized.append(w)
                    break

    # 2) collect other high-score terms excluding other bodies and redundancy
    selected = []
    used_bodies = {dom} if dom else set()
    # seed with prioritized (dedup/subsumption-aware)
    for w in prioritized:
        if not is_redundant(w, selected):
            selected.append(w)
            if any(b in find_bodies(w) for b in BODIES):
                used_bodies.update(find_bodies(w))
        if len(selected) >= top_k:
            return selected[:top_k]

    for w in words:
        # skip if term mentions a *different* body than the dominant one
        wb = set(find_bodies(w))
        if wb and (dom is None or (wb - {dom})):
            continue
        if not is_redundant(w, selected):
            selected.append(w)
            if any(b in wb for b in BODIES):
                used_bodies.update(wb)
        if len(selected) >= top_k:
            break

    # final dedup pass: drop single-word body if a longer phrase with that body is present
    final = []
    present = set([s.lower() for s in selected])
    body_only = {b for b in BODIES if b in present}
    for s in selected:
        sl = s.lower()
        if sl in body_only and any((" " in t and sl in t.lower().split()) for t in selected):
            # skip bare "mars" if we already have "mars rover", etc.
            continue
        final.append(s)
        if len(final) >= top_k:
            break
    return final

start_time = time.time()
cluster_label = {}
for cid in sorted(set(labels) - {-1}):
    docs = texts[labels == cid].tolist()
    # cluster_label[cid] = topic_words(docs, top_k=CLUSTER_LABEL_K)
    cluster_label[cid] = ", ".join(topic_words_distinct(docs, min_df=PRUNE_WORDS_MIN_DF))

end_time = time.time()
print(f"Clustering operation time: {end_time - start_time:.4f} seconds")

df["cluster_label"] = df["cluster_id"].map(cluster_label).fillna("noise/generic")

# 4) quick peek (sample titles per cluster)
for cid in sorted(set(labels) - {-1})[:5]:
    print(f"\nCluster {cid}: {cluster_label[cid]}")
    print(df.loc[df.cluster_id==cid, "title"].head(5).to_string(index=False))

# 5) save
df.to_csv(SAVE_CLUSTERED_FILE, index=False)
np.save(SAVE_CLUSTERED_LABELS, labels)