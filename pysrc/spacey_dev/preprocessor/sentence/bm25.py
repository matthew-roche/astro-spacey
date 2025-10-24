import bm25s, pandas as pd
from spacey_util.add_path import data_processed_path
import Stemmer

DF_FILE = data_processed_path() /'spacenews_sentences.parquet'
df = pd.read_parquet(DF_FILE)

corpus = df['sentence'].values

stemmer = Stemmer.Stemmer("english")

# # Tokenize the corpus and index it
corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer)
retriever = bm25s.BM25()
retriever.index(corpus_tokens)

# # Happy with your index? Save it for later...
retriever.save("./data/out/bm25s")

# def list_str(l):
#     if len(l) < 1:
#         return "0"
#     else:
#         return " ".join(l)

# df['body_test'] = df['bodies'].apply(list_str)
# df['prop_text'] = df['property'].apply(list_str)
# df['celestial'] = df.apply(lambda x: f"{x.body_test} {x.prop_text}", axis=1)

# print(df['celestial']) 

# cele_body_corpus = df['celestial'].values

# corpus_tokens = bm25s.tokenize(cele_body_corpus, stopwords="en", stemmer=stemmer)
# retriever = bm25s.BM25()
# retriever.index(corpus_tokens)

# retriever.save("./data/out/bm25s_celestial")