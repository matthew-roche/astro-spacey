import json, pandas as pd, numpy as np
from spacey_util.add_path import data_processed_path
from spacey_dev.util.helper import clean_text
from spacey_util.celestial.classifier import celestial_extract
import pysbd

PROCESSED_FILE_SAVE_PATH = data_processed_path() / "spacenews_coarse_filter3.parquet"

UPDATE_DF = False
df = pd.read_parquet(PROCESSED_FILE_SAVE_PATH)
if "id" not in df.columns:
    df['id'] = range(1, len(df) + 1)

seg = pysbd.Segmenter(language="en", clean=True)

def split_sentences_pysbd(text: str, row_id: int = 0):
    sents = seg.segment(text)  # list[str]
    out, i = [], 0
    ctx_body = []
    for s in sents:
        cpy_bodies = []
        start = text.find(s, i)     # stable forward search
        end = start + len(s)
        bodies, props = celestial_extract(s)

        if len(bodies) > 0 and not any(b in ctx_body for b in bodies):
            c = set(ctx_body + bodies)
            ctx_body = list(c)
        
        if len(props) > 0:
            cpy_bodies.extend(ctx_body)
        
        out.append({"sentence": s, "start": start, "end": end, "ctx_id": row_id, "bodies": cpy_bodies, "property": props}) # (sentence, start_char, end_char)
        i = end
    return out

data = []
for i, item in df.iterrows():
    cleaned_ctx = clean_text(item.get('content'))
    sentences = split_sentences_pysbd(cleaned_ctx, item.get('id'))
    data.extend(sentences)

df_sentences = pd.DataFrame(data)

df_sentences['id'] = range(1, len(df_sentences) + 1)
print(df_sentences)

df_sentences.to_parquet(data_processed_path() / "spacenews_sentences.parquet")
df_sentences.to_csv(data_processed_path() / "spacenews_sentences.csv")

if UPDATE_DF:
    df.to_parquet(data_processed_path() / "spacenews_coarse_filter3.parquet")
    df.to_csv(data_processed_path() / "spacenews_coarse_filter3.csv")
