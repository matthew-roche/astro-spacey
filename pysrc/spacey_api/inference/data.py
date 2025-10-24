import pandas as pd
import pyarrow.parquet as pq

SENTENCE_FILE_SAVE_PATH = "./data/out/spacenews_sentences.parquet"
RECORD_FILE_SAVE_PATH = "./data/out/spacenews_coarse_filter3.parquet"


def load_sentence_data():
    return pd.read_parquet(SENTENCE_FILE_SAVE_PATH)

def load_db_pyarrow():
    return pq.read_table(RECORD_FILE_SAVE_PATH)