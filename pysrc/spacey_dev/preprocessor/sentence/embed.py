from transformers import AutoModel, AutoTokenizer
import time, torch
import pandas as pd
import numpy as np
from spacey_util.add_path import model_path, data_post_process, data_processed_path
from spacey_dev.preprocessor.simcse.embed import batch_embed
from spacey_dev.preprocessor.simcse.faiss import build_IndexHNSWFlat

DF_FILE = data_processed_path() /'spacenews_sentences.parquet'
SAVE_TO_FILE = data_post_process() / "spacenews_sentences.npy"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_dir = model_path() / "princeton-nlp-sup-simcse-roberta-base"

start_time = time.time()
model = AutoModel.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
end_time = time.time()
print(f"Model load time: {end_time - start_time:.4f} seconds")

model.to(device)

df = pd.read_parquet(DF_FILE)

texts = df['sentence'].values

embeddings = batch_embed(tuple(texts), model, tokenizer)

np.save(SAVE_TO_FILE, embeddings)

# Embedding generation time: 24.2437 seconds

# build_IndexHNSWFlat(SAVE_TO_FILE, './data/out/spacenews_sentences.faiss')