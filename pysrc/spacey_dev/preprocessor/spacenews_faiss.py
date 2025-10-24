from spacey_util.add_path import data_path, data_processed_path
from spacey_dev.preprocessor.simcse.faiss import build_IndexHNSWFlat

EMBED_FILE_PATH = data_processed_path() / "spacenews_simcse.npy"
SAVE_FILE_PATH = str(data_processed_path()) + "/spacenews_index.faiss"

# build faiss index
build_IndexHNSWFlat(EMBED_FILE_PATH, SAVE_FILE_PATH)

