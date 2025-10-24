from spacey_dev.util.helper import clean_text, realign_answer_start
from spacey_util.add_path import data_raw_path, data_processed_path
import json, os
from datasets import load_from_disk

def map_paragraph_to_qa(row):
  qac = [] # question, answer, context
  context = clean_text(row['context'])
  for qa in row['qas']:
    question = clean_text(qa['question'])
    answers = qa['answers']
    ans_obj = {
        "text": [], "answer_start": [], "answer_end": []
    }
    for ans in answers:
      answer_text = clean_text(ans['text'])

      new_answer_start = realign_answer_start(cleaned_ctx=context, cleaned_ans=answer_text, raw_ctx=row['context'], raw_ans=ans['text'], old_start=ans['answer_start'])

      answer_end =[]
      if new_answer_start is not None:
        answer_end = new_answer_start + len(answer_text)
      # ans.append({
      #   "text": answer_text,
      #   "answer_start": new_answer_start,
      #   "answer_end": answer_end
      # })
      ans_obj['text'].append(answer_text)
      ans_obj['answer_start'].append(new_answer_start)
      ans_obj['answer_end'].append(answer_end)
    qac.append({
      "id": qa['id'],
      "question": question,
      "is_impossible": qa['is_impossible'],
      "answers": ans_obj,
      "context": context
    })

  return qac

def run(ds):
  dataset = ds.to_iterable_dataset()
  processed_ds = []
  processed_topics = {}
  for r, row in enumerate(dataset): # 1 row
     for idx, article in enumerate(row['data']):
        processed_topics[article['title']] = 0
        for j, p in enumerate(article['paragraphs']):
            if len(p['context']) < 1:
                continue # skip empty context
            qa = map_paragraph_to_qa(p)
            processed_ds.extend(qa)
            processed_topics[article['title']] += len(qa)

  return processed_ds, processed_topics

if __name__ == "__main__":
    ds = load_from_disk(data_raw_path() / "nasa-smd-qa")
    print(ds)

    rows_train = len(ds["train"])
    rows_val = len(ds["validation"])

    train_cleaned, train_titles = run(ds["train"])
    val_cleaned, val_titles = run(ds["validation"])

    print(len(train_cleaned), len(val_cleaned))

    with open(data_processed_path() / f"nasa_smd_train_cleaned_v1_alpha.json", "w", encoding="utf-8") as file:
        json.dump(train_cleaned, file, ensure_ascii=False, indent=2)
    with open(data_processed_path() / f"nasa_smd_val_cleaned_v1_alpha.json", "w", encoding="utf-8") as file:
        json.dump(val_cleaned, file, ensure_ascii=False, indent=2)
