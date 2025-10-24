import time, torch
import pandas as pd
import numpy as np

def encode_batch(text, model, tokenizer, MAX_LEN = 512, STRIDE = 128):
    tokens = tokenizer(
        text,
        max_length=MAX_LEN,
        stride=STRIDE,
        truncation=True,
        padding="max_length",
        #return_overflowing_tokens=True,
        return_tensors="pt"
    )
    tokens.pop("overflow_to_sample_mapping", None)
    tokens.pop("num_truncated_tokens", None) 

    with torch.no_grad():
        tokens_cuda = {k:v.cuda() for k,v in tokens.items() if isinstance(v, torch.Tensor)}
        out = model(**tokens_cuda, output_hidden_states=True, return_dict=True)
        if out.pooler_output is not None:
            embs = out.pooler_output
        else:
            last = out.last_hidden_state
            mask = tokens_cuda["attention_mask"].unsqueeze(-1)
            embs = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        embs = torch.nn.functional.normalize(embs, p=2, dim=1)
    return embs.cpu().numpy()
    

def batch_embed(texts, model, tokenizer, BATCH_SIZE = 64):
    all_embs = []
    start_time = time.time()
    texts_len = len(texts)
    for i in range(0, texts_len, BATCH_SIZE):
        print(f"Processing {i+BATCH_SIZE}/{texts_len}")
        batch_texts = texts[i:i+BATCH_SIZE]
        batch_embeddings = encode_batch(batch_texts, model, tokenizer)
        all_embs.append(batch_embeddings)
    
    end_time = time.time()
    all_embs = np.vstack(all_embs)
    assert all_embs.shape[0] == texts_len # generated embeddings match rows considered

    if __debug__:
        print(f"Embedding generation time: {end_time - start_time:.4f} seconds")
    
    return np.vstack(all_embs)
