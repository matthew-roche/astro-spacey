from transformers import AutoModelForQuestionAnswering, AutoModel, AutoTokenizer, pipeline
import torch, time
import bm25s, Stemmer, numpy as np

global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

stemmer = Stemmer.Stemmer("english")
bm25_retriever = bm25s.BM25.load("./data/out/bm25s")

# NQ_NASA_V1_DIR = "./models/nq_nasa_v1"
NQ_NASA_V1_DIR = "./models/quantaRoche-roberta-finetuned-nq-nasa-qa"

SIMCSE_MODEL_DIR = "./models/princeton-nlp-sup-simcse-roberta-base"

# TENSORT_ENGINE_DIR = "./models/roberta-base-squad2-nq-nasa-cp90-5.trt"

def torch_device():
    return device

def load_nq_nasa_v1_pipeline():
    return pipeline("question-answering", model=AutoModelForQuestionAnswering.from_pretrained(NQ_NASA_V1_DIR),
        tokenizer=AutoTokenizer.from_pretrained(NQ_NASA_V1_DIR), device_map=device)

# def load_tensorrt_engine():
#     with open(TENSORT_ENGINE_DIR, "rb") as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
#         engine = runtime.deserialize_cuda_engine(f.read()) 

def load_simcse_model():
    start_time = time.time()
    model = AutoModel.from_pretrained(SIMCSE_MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(SIMCSE_MODEL_DIR)
    end_time = time.time()
    if __debug__:
        print(f"SimCSE Model load time: {end_time - start_time:.4f} seconds")
    
    model.to(device).eval() # evaluation mode
    return model, tokenizer
