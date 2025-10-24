import time
import pandas as pd
from pandas import DataFrame
import numpy as np
from spacey_util.add_path import data_raw_path, data_processed_path, report_path
from spacey_dev.util.helper import clean_text


def clean_corpus(df: DataFrame, clean_cols, drop_na_cols):
    df_copy = df.copy()
    for c in clean_cols:
        df_copy[c] = df_copy[c].apply(clean_text, args=(True,)) # True for pandas df

    # drop column if empty, because no use of content extraction
    df_copy = df_copy[df_copy[drop_na_cols].str.strip() != ""]

    return df_copy

def report(corpus_df: DataFrame, cleaned_df: DataFrame, target_cols, top_k:int = 5):
    
    comp_corpus_df = corpus_df[target_cols].copy()

    before_n = len(comp_corpus_df)
    after_n = len(cleaned_df)
    dropped_n = before_n - after_n

    print(f"Total rows: {before_n}")
    print(f"Rows dropped due to n/a content: {dropped_n}")

    comp_corpus_df = comp_corpus_df.loc[cleaned_df.index] # only select indexes that is also in cleaned df, otherwise row mismatch

    # difference in before and after clean report generation
    changed_mask_per_col = (comp_corpus_df.fillna("") != cleaned_df[target_cols])
    changed_any_row = changed_mask_per_col.any(axis=1)
    num_changed_rows = int(changed_any_row.sum())

    print(f"Rows changed (any of {target_cols}): {num_changed_rows}")

    before_long = comp_corpus_df.where(changed_mask_per_col).stack().rename("before")
    after_long  = cleaned_df[target_cols].where(changed_mask_per_col).stack().rename("after")

    report  = pd.DataFrame({
        "row_id": before_long.index.get_level_values(0),  # row index directly
        "column": before_long.index.get_level_values(1),  # column name
        "before": before_long.values,
        "after":  after_long.values
    })

    report["len_delta"] = report["before"].fillna("").str.len() - report["after"].fillna("").str.len()
    report["abs_len_delta"] = report["len_delta"].abs()
    report["before_vis"] = report["before"].fillna("").apply(lambda s: s.encode("unicode_escape").decode("ascii"))

    # ensure columns sort in your preferred order
    report["column"] = pd.Categorical(report["column"], categories=target_cols, ordered=True)

    # A) Top-K cells overall (regardless of which column)
    top_cells_overall = (report
        .sort_values(["abs_len_delta","column","row_id"], ascending=[False, True, True])
        .head(top_k))

    # B) Top-K per column (e.g., top K changed "title", top K changed "content", ...)
    top_cells_per_col = (report
        .sort_values(["column","abs_len_delta","row_id"], ascending=[True,False,True])
        .groupby("column", group_keys=False)
        .head(top_k))
    
    return top_cells_overall, top_cells_per_col


if __name__ == "main":
    DATASET_NAME = "spacenews"
    DF_FILE = data_raw_path() / f'{DATASET_NAME}.csv'
    CLEANED_FILE_SAVE_PATH = data_processed_path() / f"{DATASET_NAME}.parquet"

    df = pd.read_csv(DF_FILE, encoding="utf-8")
    print(df)

    TEXT_COLS_CHECK = ["title", "content", "postexcerpt"]
    DROP_NA_COL = "content" # if there is no content then drop
    ID_COL = "id"

    # for comparison reporting purpose
    SELECT_DIFFERENCE_TOP_K = 5
    DIFFERNCE_FILE_SAVE_PATH = data_processed_path() / f"{DATASET_NAME}_clean_report.csv"

    df_copy = df[TEXT_COLS_CHECK].copy()

    print(f"Starting cleaner...")
    start_time = time.time()
    cleaned_df = clean_corpus(df, TEXT_COLS_CHECK, DROP_NA_COL)
    end_time = time.time()
    print(f"Clean time: {end_time - start_time:.4f} seconds")

    df_copy = df_copy.loc[cleaned_df.index] # df_copy select un dropped indexes


    # save cleaned dataframe
    cleaned_df.to_parquet(CLEANED_FILE_SAVE_PATH, index=False)
    print(f"Saved cleaned data: {CLEANED_FILE_SAVE_PATH}")

    top_cells_overall, top_cells_per_col = report(df_copy, cleaned_df, TEXT_COLS_CHECK, ID_COL, top_k=SELECT_DIFFERENCE_TOP_K)

    # save report
    top_cells_overall.to_csv(report_path() / f"{DATASET_NAME}_clean_overall.csv", index=False)
    top_cells_per_col.to_csv(report_path() / f"{DATASET_NAME}_clean_per_column.csv", index=False)

    print(f"Saved before/after top-K report: {DIFFERNCE_FILE_SAVE_PATH}")


# Total rows: 20716
# Rows dropped due to n/a content: 169
# Rows changed (any of ['title', 'content', 'postexcerpt']): 17468