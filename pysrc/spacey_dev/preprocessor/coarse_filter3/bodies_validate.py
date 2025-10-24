import pandas as pd, json
from spacey.add_path import data_processed_path

df = pd.read_parquet(data_processed_path() / "spacenews_coarse_filter3.parquet")

with open(data_processed_path() / "spacenews_coarse_filter3.json", "r") as f:
    classified_results = json.load(f)


select_df = df[df['content'].str.contains('Mercury')]
for i, item in df.iterrows():
    print()
    print(item['title'])
    print(item['content'])