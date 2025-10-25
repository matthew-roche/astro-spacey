def set_seed(seed: int=42):
    import random, numpy as np, torch, os
    os.environ['PTTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

set_seed(52)

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    default_data_collator
)
import evaluate, json, re
import numpy as np
import collections
from pathlib import Path
from spacey_util.add_path import fine_tuned_model_path, data_processed_path
from peft import LoraConfig, AutoPeftModelForQuestionAnswering, get_peft_model, TaskType
# from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

def project_path():
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    return project_dir

# ========= Configuration =========
USE_LORA           = False
TRAIN_JSON_PATH    = "nasa_smd_train_cleaned_v1_alpha.json"
DEV_JSON_PATH      = "nasa_smd_val_cleaned_v1_alpha.json"

MAX_SEQ_LEN        = 384
DOC_STRIDE_TOKENS  = 128
TRAIN_BATCH_SIZE   = 8
EVAL_BATCH_SIZE    = 16
LEARNING_RATE      = 3e-5
NUM_EPOCHS         = 5
EVAL_STEPS         = 10 # compute f1 em
SAVE_STEPS         = 10 # checkpoint save
LOG_STEPS          = 5 # log at each
OUTPUT_DIR         = "./finetuned_models/roberta-base-squad2-nq-nasa"
# USE_BF16           = True                            # 4080 Super supports bfloat16, but collab doesn't
USE_GRAD_CHKPT     = True                            # memory saver on long seqs
base_model_dir     = fine_tuned_model_path() / "roberta-base-squad2-nq"

# ========= Load data & model =========
dataset = load_dataset("json", data_files={"train": TRAIN_JSON_PATH, "validation": DEV_JSON_PATH}, data_dir=data_processed_path())
tokenizer = AutoTokenizer.from_pretrained(base_model_dir)

# ========= Trainable Params ==========

TARGET_LAST_K_LAYERS = 12
NUM_OF_LAYERS = 12

if USE_LORA:
    # LoRA, https://github.com/huggingface/peft/tree/v0.17.0
    LORA_R = 128
    LORA_ALPHA = 256
    LORA_DROPOUT = 0.05
    TARGET_MODULES = [
        "query", # changes what the token looks for in other tokens
        # "key", # changes which tokes are picked often
        "value", # info update after attention
    ]

    # lora peft configuration
    peft_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.QUESTION_ANS,
        target_modules=TARGET_MODULES,
        layers_to_transform=list(range(0, 2)),
        layers_pattern="layer" # layer.#
    )

    checkpoint_model = AutoModelForQuestionAnswering.from_pretrained(base_model_dir)
    model = get_peft_model(checkpoint_model, peft_cfg)
    model.print_trainable_parameters()

else:
    # method 2
    model = AutoModelForQuestionAnswering.from_pretrained(base_model_dir)
    def is_lastK_layer(n): 
        return any(re.search(fr"encoder\.layer\.{i}\.", n) for i in range(0, 12))


    head_params, lastk_params = [], []
    for n,p in model.named_parameters():
        print (n)
        if "qa_outputs" in n:
            p.requires_grad = True; 
            head_params.append(p)
        #elif is_lastK_layer(n) and ("attention" in n or "LayerNorm" in n or "dense" in n):
        # elif "embeddings.word_embeddings" in n:
        #     p.requires_grad = True; 
        #     lastk_params.append(p)
        else:
        #     p.requires_grad = False
            lastk_params.append(p)

    optimizer = torch.optim.AdamW([
        {"params": head_params, "lr": 1e-5},
        {"params": lastk_params, "lr": 1e-6},
    ], weight_decay=0.01)

    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters:     {total_params:,}")
    print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")

# show trainable layers for debug
# for n,p in model.named_parameters():
#     if not p.requires_grad:
#         print(n)

# ========= Preprocessing (train/eval) =========
# This mirrors the SQuAD2 recipe: sliding window over context and label start/end token positions.
QUESTION_ON_LEFT_CONTEXT_ON_RIGHT = True  # RoBERTa convention used below

def build_train_features(examples):
    """Tokenize question+context with a sliding window and compute start/end labels."""
    questions = [q.strip() for q in examples["question"]]
    tokenized = tokenizer(
        questions,
        examples["context"],
        truncation="only_second" if QUESTION_ON_LEFT_CONTEXT_ON_RIGHT else "only_first",
        max_length=MAX_SEQ_LEN,
        stride=DOC_STRIDE_TOKENS,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,   # char offsets per token for span alignment
        padding="max_length",
    )

    # Maps each windowed feature back to the original example index
    example_index_for_feature = tokenized.pop("overflow_to_sample_mapping")
    token_char_offsets = tokenized.pop("offset_mapping")

    start_positions, end_positions = [], []

    for feature_idx, offsets_for_feature in enumerate(token_char_offsets):
        example_idx = example_index_for_feature[feature_idx]
        answer_info = examples["answers"][example_idx]

        # Handle unanswerable examples
        if len(answer_info["answer_start"]) == 0:
            start_positions.append(0)
            end_positions.append(0)
            continue

        answer_start_char = answer_info["answer_start"][0]
        answer_end_char   = answer_start_char + len(answer_info["text"][0])

        # Find token span that corresponds to the context segment in this window
        sequence_ids = tokenized.sequence_ids(feature_idx)
        # Locate first token of the context in this feature
        tok_idx = 0
        while tok_idx < len(sequence_ids) and sequence_ids[tok_idx] != (1 if QUESTION_ON_LEFT_CONTEXT_ON_RIGHT else 0):
            tok_idx += 1
        context_first_tok = tok_idx
        # Locate last token of the context in this feature
        while tok_idx < len(sequence_ids) and sequence_ids[tok_idx] == (1 if QUESTION_ON_LEFT_CONTEXT_ON_RIGHT else 0):
            tok_idx += 1
        context_last_tok = tok_idx - 1

        # If the answer is not fully inside this window, mark as impossible for this feature
        if not (offsets_for_feature[context_first_tok][0] <= answer_start_char
                and offsets_for_feature[context_last_tok][1] >= answer_end_char):
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Move start token pointer until it passes answer_start_char
            start_tok = context_first_tok
            while start_tok <= context_last_tok and offsets_for_feature[start_tok][0] <= answer_start_char:
                start_tok += 1
            # Move end token pointer backward until it is before answer_end_char
            end_tok = context_last_tok
            while end_tok >= context_first_tok and offsets_for_feature[end_tok][1] >= answer_end_char:
                end_tok -= 1

            start_positions.append(start_tok - 1)
            end_positions.append(end_tok + 1)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"]   = end_positions
    return tokenized


def build_eval_features(examples):
    """Tokenize question+context with sliding window and compute start/end labels + keep offsets/example_id."""
    questions = [q.strip() for q in examples["question"]]
    tokenized = tokenizer(
        questions,
        examples["context"],
        truncation="only_second" if QUESTION_ON_LEFT_CONTEXT_ON_RIGHT else "only_first",
        max_length=MAX_SEQ_LEN,
        stride=DOC_STRIDE_TOKENS,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    example_index_for_feature = tokenized.pop("overflow_to_sample_mapping")
    token_char_offsets = tokenized["offset_mapping"]  # don’t pop; we’ll still need it for decoding
    # tokenized["example_id"] = [examples["id"][idx] for idx in example_index_for_feature]
    tokenized["example_id"] = [str(examples["id"][idx]) for idx in example_index_for_feature] # update

    start_positions, end_positions = [], []

    for feat_idx, offsets_for_feature in enumerate(token_char_offsets):
        ex_idx = example_index_for_feature[feat_idx]
        ans = examples["answers"][ex_idx]

        # figure out which side is context in sequence_ids
        context_flag = 1 if QUESTION_ON_LEFT_CONTEXT_ON_RIGHT else 0
        seq_ids = tokenized.sequence_ids(feat_idx)

        # context token span
        i = 0
        while i < len(seq_ids) and seq_ids[i] != context_flag:
            i += 1
        ctx_start = i
        while i < len(seq_ids) and seq_ids[i] == context_flag:
            i += 1
        ctx_end = i - 1

        if len(ans["answer_start"]) == 0:
            # no-answer → set both to CLS token index (for RoBERTa <s> is usually at pos 0)
            cls_idx = 0
            start_positions.append(cls_idx)
            end_positions.append(cls_idx)
        else:
            start_char = ans["answer_start"][0]
            end_char = start_char + len(ans["text"][0])

            # if answer not fully inside this window → mark impossible for this feature
            if not (offsets_for_feature[ctx_start][0] <= start_char and offsets_for_feature[ctx_end][1] >= end_char):
                cls_idx = 0
                start_positions.append(cls_idx)
                end_positions.append(cls_idx)
            else:
                # walk to token boundaries
                s_tok = ctx_start
                while s_tok <= ctx_end and offsets_for_feature[s_tok][0] <= start_char:
                    s_tok += 1
                e_tok = ctx_end
                while e_tok >= ctx_start and offsets_for_feature[e_tok][1] >= end_char:
                    e_tok -= 1
                start_positions.append(s_tok - 1)
                end_positions.append(e_tok + 1)

        # keep offsets only for context tokens to aid decoding; null out others
        offsets = tokenized["offset_mapping"][feat_idx]
        tokenized["offset_mapping"][feat_idx] = [
            (o if seq_ids[t] == context_flag else None) for t, o in enumerate(offsets)
        ]

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"]   = end_positions
    return tokenized


train_dataset = dataset["train"].map(
    build_train_features,
    batched=True,
    remove_columns=dataset["train"].column_names,
    desc="Tokenizing + labeling (train)",
)

eval_examples = dataset["validation"]
eval_features = eval_examples.map(
    build_eval_features,
    batched=True,
    remove_columns=eval_examples.column_names,
    desc="Tokenizing (eval)",
)

# ========= Metrics =========
# Computes EM/F1 using the standard SQuAD v2 metric after converting logits to text spans.
squad_v2_metric = evaluate.load("squad_v2")

def postprocess_logits_to_text(examples, features, predictions, n_best=20, max_answer_len=30, handle_impossible:bool=True):
    start_logits, end_logits = predictions
    example_to_feature_indices = collections.defaultdict(list)
    for feat_idx, feat in enumerate(features):
        example_to_feature_indices[feat["example_id"]].append(feat_idx)

    final_pred_text = {}
    for ex in examples:
        ex_id = ex["id"]
        ctx_text = ex["context"]
        candidate_spans = []

        for feat_idx in example_to_feature_indices[ex_id]:
            starts = start_logits[feat_idx]
            ends   = end_logits[feat_idx]
            offsets = features[feat_idx]["offset_mapping"]

            top_start_idxs = np.argsort(starts)[-n_best:][::-1]
            top_end_idxs   = np.argsort(ends)[-n_best:][::-1]

            for s in top_start_idxs:
                for e in top_end_idxs:
                    if e < s or (e - s + 1) > max_answer_len:
                        continue
                    if offsets[s] is None or offsets[e] is None:
                        continue
                    s_char, e_char = offsets[s][0], offsets[e][1]
                    candidate_spans.append(((starts[s] + ends[e]).item(), ctx_text[s_char:e_char]))

        # Best scoring span; empty string becomes "no-answer"
        final_pred_text[ex_id] = "" if not candidate_spans else sorted(candidate_spans, key=lambda x: x[0], reverse=True)[0][1]
        # update
        if handle_impossible:            
            null_score = max( (start_logits[fi][0] + end_logits[fi][0]) for fi in example_to_feature_indices[str(ex_id)] )
            best_non_null_score, best_text = max(candidate_spans, default=(-1e30, ""))
            final_pred_text[ex_id] = "" if (null_score > best_non_null_score) else best_text

    references = []
    for ex in examples:
        ex_id = str(ex["id"])
        ans = ex["answers"]
        if not ans or len(ans.get("text", [])) == 0:
            references.append({"id": ex_id, "answers": {"text": [""], "answer_start": [0]}})
        else:
            references.append({
                "id": ex_id,
                "answers": {
                    "text": list(ans["text"]),
                    "answer_start": [int(s) for s in ans["answer_start"]],
                },
            })
    return final_pred_text, references

def compute_em_f1(eval_pred):
    # eval_pred.predictions == (start_logits_np, end_logits_np)
    start_np, end_np = eval_pred.predictions

    # Your existing function that turns logits -> {"id": text} and refs list
    pred_texts, refs = postprocess_logits_to_text(
        eval_examples,           # raw examples (ids, contexts, answers)
        eval_features,           # tokenized features you passed as eval_dataset
        (start_np, end_np),      # logits
    )

    preds = [
        {"id": k, "prediction_text": v, "no_answer_probability": 1.0 if v == "" else 0.0}
        for k, v in pred_texts.items()
    ]

    out = squad_v2_metric.compute(predictions=preds, references=refs)
    # Return plain floats; Trainer will log them as eval_f1 / eval_exact
    return {
        "f1": float(out["f1"]),
        "exact": float(out["exact"]),
        "HasAns_f1": float(out.get("HasAns_f1", 0.0)),
        "NoAns_f1": float(out.get("NoAns_f1", 0.0)),
    }

print("eval_features: ", len(eval_features))

def prep_logits_for_metrics(logits, labels):
    # logits is a tuple from the QA head
    start, end = logits
    # Convert to CPU numpy for cheap post-processing
    return (start, end)

# ========= Training =========
train_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    num_train_epochs=NUM_EPOCHS,
    eval_strategy="steps",             # or "steps"
    eval_steps = EVAL_STEPS,
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",
    greater_is_better=True,

    eval_accumulation_steps=4,         # accumulate eval batches on GPU before gathering
    save_strategy="steps",             # "steps" or "epoch"
    save_steps=SAVE_STEPS,
    logging_strategy="steps",
    logging_steps=LOG_STEPS,
    gradient_checkpointing=USE_GRAD_CHKPT,
    # tf32=USE_TF32, # port to collab later
    report_to="none",
    label_names=["start_positions", "end_positions"] 
)

trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=train_dataset,
    eval_dataset=eval_features,
    tokenizer=tokenizer,
    data_collator=default_data_collator,
    compute_metrics=compute_em_f1,
    preprocess_logits_for_metrics=prep_logits_for_metrics,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Finished. Model saved to: {OUTPUT_DIR}")

with open(f"{OUTPUT_DIR}/log_history.json", "w") as f:
    json.dump(trainer.state.log_history, f, ensure_ascii=False, indent=2)

# 1828/1828 [09:22<00:00], len(eval_features) = 905

if USE_LORA:
    # merge
    dense_out = OUTPUT_DIR + "-merged"
    print("[Merge] Loading base+adapters for merge...")
    merged = AutoPeftModelForQuestionAnswering.from_pretrained(
        OUTPUT_DIR,
        device_map="cpu",
        # torch_dtype=torch.bfloat16 if BF16 else torch.float16,
    )
    merged = merged.merge_and_unload()
    merged.save_pretrained(dense_out, safe_serialization=True)
    tokenizer.save_pretrained(dense_out)
    print(f"[Merge] Merged dense model saved to: {dense_out}")
