import time, torch
import faiss, numpy as np, pandas as pd, re

def build_IndexHNSWFlat(EMBED_FILE_PATH, SAVE_FILE_PATH):
    #embeddings = np.load(data_path() / "spacenews-simcse.npy").astype("float32")
    embeddings = np.load(EMBED_FILE_PATH).astype("float32")
    # assume already L2-normalized; if not:
    embeddings /= (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)

    dim = embeddings.shape[1]

    if __debug__:
        print(embeddings.shape)

    index = faiss.IndexHNSWFlat(dim, 32)
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = 128
    index.add(embeddings)

    # faiss.write_index(index, "./ait500g4/data/spacenews-simcse-idx.faiss")
    faiss.write_index(index, SAVE_FILE_PATH)


