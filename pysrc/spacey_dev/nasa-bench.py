from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import time, torch, random, json
from evaluate import load as load_metric
from spacey_util.add_path import data_processed_path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BENCH_NQ_MODEL = False
BENCH_NQ_NASA_MODEL = True

NASA_SMD_VALIDATION_FILE_PATH = data_processed_path() / "nasa_smd_val_cleaned_v1_alpha.json"

with open(NASA_SMD_VALIDATION_FILE_PATH, "r", encoding="utf-8") as f:
    validation_list = json.load(f)

SAVE_MODEL_DIR = "./models/deepset-roberta-base-squad2"
SAVE_T_DIR = "./models/deepset-roBerta-base-squad2-token"

roberta_base_sq2_pipeline = pipeline("question-answering", 
                                     model=AutoModelForQuestionAnswering.from_pretrained(SAVE_MODEL_DIR),
                                     tokenizer=AutoTokenizer.from_pretrained(SAVE_T_DIR), device_map=device)

if BENCH_NQ_MODEL:
    FT_MODEL_DIR = "./finetuned_models/roberta-base-squad2-nq"


    roberta_base_sq2_nq_pipeline = pipeline("question-answering", 
                                        model=AutoModelForQuestionAnswering.from_pretrained(FT_MODEL_DIR),
                                        tokenizer=AutoTokenizer.from_pretrained(FT_MODEL_DIR), device_map=device)

if BENCH_NQ_NASA_MODEL:
    # NASA_FT_MODEL_DIR = "./finetuned_models/roberta-base-squad2-nq-nasa/checkpoint-90"
    NASA_FT_MODEL_DIR = "./models/quantaRoche-roberta-finetuned-nq-nasa-qa"


    roberta_base_sq2_nq_nasa_pipeline = pipeline("question-answering", 
                                        model=AutoModelForQuestionAnswering.from_pretrained(NASA_FT_MODEL_DIR),
                                        tokenizer=AutoTokenizer.from_pretrained(NASA_FT_MODEL_DIR), device_map=device)

ref = []
# before finetune
pred_bf_ft = []
# after finetune
pred_af_ft = []

pred_af_ft_nasa = []
for v in validation_list:
    inputs = {
        'question': v['question'],
        'context': v['context']
    }

    roberta_base_sq2_result = roberta_base_sq2_pipeline(question=inputs['question'], context=inputs['context'], handle_impossible_answer=True)

    pred_no_ans = 1.0 if roberta_base_sq2_result["answer"] == "" else 0.0
    pred_bf_ft.append({
        "id": v["id"],
        "prediction_text": roberta_base_sq2_result["answer"],
        "no_answer_probability": pred_no_ans
    })
    ref.append({
        "id": v["id"],
        "answers": {
            "text": v['answers']['text'],
            "answer_start": v['answers']['answer_start']
        }
    })

    if BENCH_NQ_MODEL:
        roberta_base_sq2_nq_result = roberta_base_sq2_nq_pipeline(question=inputs['question'], context=inputs['context'], handle_impossible_answer=True)
        pred_ft_no_ans = 1.0 if roberta_base_sq2_nq_result["answer"] == "" else 0.0
        pred_af_ft.append({
            "id": v["id"],
            "prediction_text": roberta_base_sq2_nq_result["answer"],
            "no_answer_probability": pred_ft_no_ans
        })
    
    if BENCH_NQ_NASA_MODEL:
        roberta_base_sq2_nq_nasa_result = roberta_base_sq2_nq_nasa_pipeline(question=inputs['question'], context=inputs['context'], handle_impossible_answer=True)
        pred_ft_nasa_no_ans = 1.0 if roberta_base_sq2_nq_nasa_result["answer"] == "" else 0.0
        pred_af_ft_nasa.append({
            "id": v["id"],
            "prediction_text": roberta_base_sq2_nq_nasa_result["answer"],
            "no_answer_probability": pred_ft_nasa_no_ans
        })

    #   if v['is_impossible'] and roberta_base_sq2_result["answer"] != roberta_base_sq2_nq_result["answer"]:
    #     print("\nQ:", v['question'])
    #     print("before_finetune:", roberta_base_sq2_result["answer"])
    #     print("after_finetune:", roberta_base_sq2_nq_result["answer"])


metric = load_metric("squad_v2")

before_finetune_scores = metric.compute(predictions=pred_bf_ft, references=ref)
# print(f"before_finetune_scores, f1 {before_finetune_scores['f1']}, HasAns_f1: {before_finetune_scores['HasAns_f1']}, EM: {before_finetune_scores['HasAns_exact']}") # f1 is weighted, take has ans f1
print(before_finetune_scores)

if BENCH_NQ_MODEL:
    after_finetune_scores = metric.compute(predictions=pred_af_ft, references=ref)
    # print(f"after_finetune_scores, f1 {after_finetune_scores['f1']}, HasAns_f1: {after_finetune_scores['HasAns_f1']}, EM: {after_finetune_scores['HasAns_exact']}")
    print(after_finetune_scores)


if BENCH_NQ_NASA_MODEL:
    after_finetune_scores = metric.compute(predictions=pred_af_ft_nasa, references=ref)
    # print(f"after_finetune_scores, f1 {after_finetune_scores['f1']}, HasAns_f1: {after_finetune_scores['HasAns_f1']}, EM: {after_finetune_scores['HasAns_exact']}")
    print(after_finetune_scores)
