import ftfy, unicodedata, re, numpy as np, pandas as pd

def realign_answer_start(cleaned_ctx: str, raw_ctx:str, cleaned_ans: str, raw_ans: str, old_start: int | None = None):
    # 1) exact matches
    exact_hits_old = [m.start() for m in re.finditer(rf"{re.escape(raw_ans)}", raw_ctx)]
    exact_hits = [m.start() for m in re.finditer(rf"{re.escape(cleaned_ans)}", cleaned_ctx)]

    if exact_hits:
        if old_start is None or old_start not in exact_hits_old:
            return exact_hits[0]
        
        if len(exact_hits_old) == len(exact_hits) and old_start in exact_hits_old:
            index_of_old_start_rel_old_hits = exact_hits_old.index(old_start)

            return exact_hits[index_of_old_start_rel_old_hits]
    
    # 2) whitespace-tolerant (collapse spaces in answer -> \s+ in pattern)
    pat = re.escape(re.sub(r"\s+", " ", cleaned_ans)).replace(r"\ ", r"\s+")
    m = re.search(pat, cleaned_ctx, flags=re.IGNORECASE)
    if m:
        return m.start()

    return None


def clean_text(s, pd_na: bool = False):
  # fix na
  if pd_na:
      if pd.isna(s): return ""
  elif s is None: return ""
  
  # fix odd unicode
  s = ftfy.fix_text(str(s))
  # decompose by compatibility, recompose by canonical form
  s = unicodedata.normalize("NFKC", str(s))
  # remove carriage ret/new line
  s = s.replace("\r\n"," ").replace("\n"," ").replace("\\n"," ")
  s = "".join(" " if ord(c)<32 or ord(c)==127 else c for c in s)
  return " ".join(s.split())

def assert_embedding_close(ref, got, atol=3e-7, cos_min=0.9999995):
    ref = ref.astype(np.float32, copy=False)
    got = got.astype(np.float32, copy=False)

    # 1) elementwise absolute tolerance (ignore rtol; small values blow it up)
    np.testing.assert_allclose(got, ref, rtol=0.0, atol=atol)

    # 2) cosine similarity for scale-invariant agreement
    r = ref / np.linalg.norm(ref, axis=1, keepdims=True)
    g = got / np.linalg.norm(got, axis=1, keepdims=True)
    cos = float((r * g).sum())
    assert cos >= cos_min, f"cosine too low: {cos:.9f} < {cos_min}"