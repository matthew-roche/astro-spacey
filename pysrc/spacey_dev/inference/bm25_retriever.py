import bm25s, Stemmer, numpy as np
stemmer = Stemmer.Stemmer("english")

def bm25_retriever():
    return bm25s.BM25.load("./data/out/bm25s")

def retrieve_lexical_ids(retriever, query, top_k: int  = 10):
    query_tokens = bm25s.tokenize(query, stemmer=stemmer)
    results, scores = retriever.retrieve(query_tokens, k=top_k)
    
    return results[0], scores[0]


