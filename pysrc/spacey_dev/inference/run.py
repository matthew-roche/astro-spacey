import pandas as pd, torch, time
from spacey_util.add_path import data_post_process
from spacey_dev.inference.hybrid_retriever import load_simcse_model, fused_retriever
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROCESSED_FILE_SAVE_PATH = data_post_process() / "spacenews_sentences.parquet"

df = pd.read_parquet(PROCESSED_FILE_SAVE_PATH)

# df_cele_body = df['bodies']
# df_cele_prop = df['property']

simcse_model, simcse_tokenizer = load_simcse_model()

# question = "What is in oxygen-16 ?"
question = "What's on Mercury craters ?"
# question = "Which planets have evidence of water?"
# question = "Which planet has a higher and upper atmosphere ?"

# question = "Which planet has a higher atmosphere ?"

# question = "What is the total population in Mars ?"
# question = "Where is Mars ?"
# question = "Which planet has a higher and upper atmosphere ?"


# ids, scores = hybrid_retrieve(question)
start_time = time.perf_counter()
ids, scores = fused_retriever(simcse_model, simcse_tokenizer, question, df)
# ids, _ = hybrid_retrieve(question)

end_time = time.perf_counter()
time_elapsed_ms = (end_time - start_time) * 1000
print(f"Retrieval time: {time_elapsed_ms:.4f} ms")

text_list = df.iloc[ids]['sentence'].values

# for id, score, text in zip(ids, scores, text_list):
#     print(f"{id}, {score} {text}")

context = " ".join(text_list)

# FT_MODEL_DIR = "./finetuned_models/roberta-base-squad2-nq-best"
# FT_MODEL_DIR = "./finetuned_models/roberta-base-squad2-nq-nasa/checkpoint-90" # same as quantaRoche/roberta-finetuned-nq-nasa-qa
FT_MODEL_DIR = "./models/quantaRoche-roberta-finetuned-nq-nasa-qa"


pipe = pipeline("question-answering", 
                                     model=AutoModelForQuestionAnswering.from_pretrained(FT_MODEL_DIR),
                                     tokenizer=AutoTokenizer.from_pretrained(FT_MODEL_DIR), device_map=device)

start_time = time.perf_counter()
out = pipe(question=question, context=context, handle_impossible_answer=True)
end_time = time.perf_counter()
time_elapsed_ms = (end_time - start_time) * 1000
# print(f"Model Inference time: {time_elapsed_ms:.4f} ms")
# print(f"A: {out["answer"]}")
print(out)