from spacey_dev.preprocessor.nq import run as nq_pipeline
from spacey_dev.preprocessor.nasa import run as nasa_pipeline
from spacey_dev.preprocessor.spacenews_clean_text import clean_corpus, report

from spacey_util.add_path import data_raw_path, data_processed_path, report_path, data_post_process
from datasets import load_from_disk
import json, pandas as pd, time

def preprocess_save(pipe, train_split, val_split, name:str = "ds", version:str = "v1_alpha"):
    train_cleaned, t_no_ans_ids = pipe(train_split)
    val_cleaned, v_no_ans = pipe(val_split)

    with open(data_processed_path() / f"{name}_train_cleaned_{version}.json", "w", encoding="utf-8") as file:
        json.dump(train_cleaned, file, ensure_ascii=False, indent=2)
    with open(data_processed_path() / f"{name}_val_cleaned_{version}.json", "w", encoding="utf-8") as file:
        json.dump(val_cleaned, file, ensure_ascii=False, indent=2)
    
    print(len(train_cleaned), len(val_cleaned))

# preprocessor pipeline to filter and extract the files
if __name__ == "__main__":
    # nq for model training
    print("Starting preprocessor on Natural Questions Dataset...")
    ds = load_from_disk(data_raw_path() / "cjlovering-qa")
    print(ds)

    preprocess_save(nq_pipeline, ds['train'], ds['validation'], 'nq')
    
    # nasa for finetune #2 and bechnmarking
    print("Starting preprocessor on NASA SMD Dataset...")
    ds = load_from_disk(data_raw_path() / "nasa-smd-qa")
    print(ds)

    preprocess_save(nasa_pipeline, ds['train'], ds['validation'], 'nasa_smd')

    ds = None # unalloc

    # spacenews
    DATASET_NAME = "spacenews"
    DF_FILE = data_raw_path() / f'{DATASET_NAME}.csv'
    CLEANED_FILE_SAVE_PATH = data_processed_path() / f"{DATASET_NAME}.parquet"

    corpus_df = pd.read_csv(DF_FILE, encoding="utf-8")
    print(corpus_df)

    TEXT_COLS_CHECK = ["title", "content", "postexcerpt"]
    DROP_NA_COL = "content" # if there is no content then drop
    ID_COL = "id"

    # for comparison reporting purpose
    SELECT_DIFFERENCE_TOP_K = 5
    DIFFERNCE_FILE_SAVE_PATH = data_processed_path() / f"{DATASET_NAME}_clean_report.csv"

    print(f"Starting cleaner...")
    start_time = time.time()
    cleaned_df = clean_corpus(corpus_df, TEXT_COLS_CHECK, DROP_NA_COL)
    end_time = time.time()
    print(f"Clean time: {end_time - start_time:.4f} seconds")

    # # save cleaned dataframe
    cleaned_df.to_parquet(CLEANED_FILE_SAVE_PATH, index=False)
    print(f"Saved cleaned data: {CLEANED_FILE_SAVE_PATH}")

    top_cells_overall, top_cells_per_col = report(corpus_df, cleaned_df, TEXT_COLS_CHECK, top_k=SELECT_DIFFERENCE_TOP_K)

    # save report
    top_cells_overall.to_csv(report_path() / f"{DATASET_NAME}_clean_overall.csv", index=False)
    top_cells_per_col.to_csv(report_path() / f"{DATASET_NAME}_clean_per_column.csv", index=False)

    print(f"Saved before/after top-K report: {DIFFERNCE_FILE_SAVE_PATH}")

    corpus_df = None # unalloc
    cleaned_df = None # unalloc

    # if umap + hdbscan is needed, else keep commented
    # # embedding generate with simcse
    # import spacey_dev.preprocessor.spacenews_embed
    # # create faiss index
    # import spacey_dev.preprocessor.spacenews_faiss

    print("Continuing preprocessor from prior LLaMA reports...")

    # From LLaMA run results, iteration 1 extraction
    # convert LLaMA extraction report to proper json mappable values
    import spacey_dev.preprocessor.coarse_filter2.spacenews_classify_post
    import spacey_dev.preprocessor.coarse_filter2.spacenews_extractv2


    # From LLaMA run results, iteration 2 extraction
    import spacey_dev.preprocessor.coarse_filter3.bodies_classify_post

    print("Continuing preprocessor last step...")
    # sentence segmentation
    import spacey_dev.preprocessor.sentence.segment
    # create simcse embedding for segmented sentences
    import spacey_dev.preprocessor.sentence.embed
    # build faiss index
    import spacey_dev.preprocessor.sentence.build_faiss
    # build bm25 index
    import spacey_dev.preprocessor.sentence.bm25

    # copy final files to data out
    df = pd.read_parquet(data_processed_path() / f"{DATASET_NAME}.parquet")
    df.to_parquet(data_post_process() / f"{DATASET_NAME}.parquet")

    df = pd.read_parquet(data_processed_path() / f"{DATASET_NAME}_sentences.parquet")
    df.to_parquet(data_post_process() / f"{DATASET_NAME}_sentences.parquet")

    