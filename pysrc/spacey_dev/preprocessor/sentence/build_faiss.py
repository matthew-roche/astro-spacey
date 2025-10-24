import faiss, numpy as np
from spacey_util.add_path import data_post_process
from spacey_dev.preprocessor.simcse.faiss import build_IndexHNSWFlat

EMBED_FILE_PATH = data_post_process() / "spacenews_sentences.npy"

embeddings = np.load(EMBED_FILE_PATH).astype("float32")
# assume already L2-normalized; if not:
embeddings /= (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)

X = embeddings.astype('float32')
d = X.shape[1]
index = faiss.IndexFlatL2(d) # Euclidean (squared L2)
index.add(X)
faiss.write_index(index, './data/out/spacenews_sentences.faiss')

# build_IndexHNSWFlat(EMBED_FILE_PATH, './data/out/spacenews_sentences.faiss')