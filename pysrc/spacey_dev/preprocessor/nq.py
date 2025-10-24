from datasets import load_from_disk
from spacey_dev.util.helper import clean_text, realign_answer_start
from spacey_util.add_path import data_raw_path, data_processed_path
import json

def map_row_to_qa(row, q_idx: int = 0):
  question_text = clean_text(row["questions"][q_idx]["input_text"])
  context = clean_text(row["contexts"])

  # pick candidate answer if available
  ans_list = row.get("answers", [])
  candidate_answer = None
  if ans_list:
      candidate_answer = next((a for a in ans_list if a.get("input_text","").lower()=="short"), ans_list[0])

  if candidate_answer:
    answers = {"text": [], "answer_start": [], "answer_end": []}

    answer_text = clean_text(candidate_answer["span_text"])

    new_answer_start = None # default no answer
    if len(answer_text.rstrip().split()) > 0:
       new_answer_start = realign_answer_start(cleaned_ctx=context, raw_ctx=row["contexts"], cleaned_ans=answer_text, raw_ans=candidate_answer["span_text"], old_start=candidate_answer['span_start'])

    #print(candidate_answer['span_start'], new_answer_start)
    if new_answer_start is not None:
      answers["text"].append(answer_text)
      answers["answer_start"].append(new_answer_start)
      answers['answer_end'].append(new_answer_start + len(answer_text))

  else:
      answers = {"text": [], "answer_start": [], "answer_end": []}

  return {"id": row['id'], "question": question_text, "context": context, "answers": answers}, True if new_answer_start is None else False


def run(split):
    processed_ds = []
    no_ans_ids = []
    for idx, v in enumerate(split.to_iterable_dataset()):
        if not v['has_correct_context'] or len(v['contexts']) < 1:
            continue
        qa, no_ans = map_row_to_qa(v)
        
        if no_ans:
            no_ans_ids.append(v['id'])
        
        # drop, requires thorough analysis
        else:
            processed_ds.append(qa)
    return processed_ds, no_ans_ids


if __name__ == "__main__":
    ds = load_from_disk(data_raw_path() / "cjlovering-qa")

    print(ds)
    train_cleaned, t_no_ans_ids = run(ds['train'])
    val_cleaned, v_no_ans = run(ds['validation'])

    # select_ds = ds['train'].filter(lambda x: x["id"] in ['2972037497414613725'])
    # print(select_ds[0])

    # print(t_no_ans_ids)
    # print(v_no_ans)

    print(len(train_cleaned), len(val_cleaned))

    with open(data_processed_path() / f"nq_train_cleaned_v1_alpha.json", "w", encoding="utf-8") as file:
        json.dump(train_cleaned, file, ensure_ascii=False, indent=2)
    with open(data_processed_path() / f"nq_val_cleaned_v1_alpha.json", "w", encoding="utf-8") as file:
        json.dump(val_cleaned, file, ensure_ascii=False, indent=2)
