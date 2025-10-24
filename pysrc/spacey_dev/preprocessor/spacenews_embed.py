from transformers import AutoModel, AutoTokenizer
import time, torch
import pandas as pd
import numpy as np
# refactor
from spacey_util.add_path import model_path, data_path, data_processed_path
from spacey_dev.preprocessor.simcse.embed import batch_embed

DF_FILE = data_processed_path() /'spacenews.parquet'
SAVE_TO_FILE = data_processed_path() / "spacenews_simcse.npy"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_dir = model_path() / "princeton-nlp-sup-simcse-roberta-base"

start_time = time.time()
model = AutoModel.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
end_time = time.time()
print(f"Model load time: {end_time - start_time:.4f} seconds")

model.to(device)

df = pd.read_parquet(DF_FILE)

print(df)
titles = df["title"].fillna("")
excerpts = df.get("postexcerpt", pd.Series([""]*len(df))).fillna("")
contents = df.get("content", pd.Series([""]*len(df))).fillna("")

MIN_TOKENS = 8
def build_text(title, abstract, context):
    # prefer excerpt if it’s substantive; else use head of content; else title
    if len(tokenizer.tokenize(context)) >= MIN_TOKENS:
        return f"{context}", False
    # elif len(tokenizer.tokenize(context)) >= MIN_TOKENS:
    #     # take head; we’ll stride if needed
    #     return f"{title} — {context}", True
    # elif len(tokenizer.tokenize(title)) >= 1:
    #     return title, False
    else:
        return "", False

pairs = [build_text(t, e, c) for t, e, c in zip(titles, excerpts, contents)]
texts  = [p[0] for p in pairs]
# use_stride = [p[1] for p in pairs]

content_present_idx = []
empty_idx = []
for idx, x in enumerate(texts):
    if x.strip():
        content_present_idx.append(idx)
    else:
        empty_idx.append(idx)
print(f"Out of {df.count}, empty: {len(empty_idx)}, has_content: {len(content_present_idx)}")

texts = [texts[i] for i in content_present_idx]
# use_stride = [use_stride[i] for i in content_present_idx]

embeddings = batch_embed(texts, model, tokenizer)

np.save(SAVE_TO_FILE, embeddings)

# Roberta-base Embedding generation time: 112.3292 seconds
# Roberta-large Embedding generation time: 300.0723 seconds