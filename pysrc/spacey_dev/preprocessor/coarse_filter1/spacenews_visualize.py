import umap, matplotlib.pyplot as plt
from spacey_util.add_path import data_path, data_processed_path
import numpy as np, pandas as pd, umap, hdbscan

df = pd.read_csv(data_processed_path() / "spacenews_clustered.csv")

def shorten(label, n=2):
    return ", ".join(label.split(", ")[:n])  # just first 2 words

df["short_label"] = df["cluster_label"].apply(lambda x: shorten(str(x), 2))

X = np.load(data_processed_path() / "spacenews_simcse.npy")    # embeddings you clustered with

# reconstruct
um2 = umap.UMAP(n_neighbors=60, min_dist=0.05, n_components=50,
               metric="cosine", random_state=42).fit_transform(X)

df["x"], df["y"] = um2[:,0], um2[:,1]

print(f"Number of rows labelled as noise: {np.sum(df['cluster_id']==-1)}")

plt.figure(figsize=(10,8))
for cid in sorted(df["cluster_id"].unique()):
    mask = df["cluster_id"]==cid
    plt.scatter(df.loc[mask,"x"], df.loc[mask,"y"], s=5,
                label=(f"{cid}: {df.loc[mask,'short_label'].iloc[0][:30]}…"
                       if cid!=-1 else "noise"), alpha=0.7)

#plt.title("UMAP of SimCSE embeddings (HDBSCAN clusters)")
# plt.axis("off")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")

plt.rcParams.update({
    'font.size': 12,          # base font size
    'axes.titlesize': 14,     # title
    'axes.labelsize': 12,     # x/y labels
    'xtick.labelsize': 10,    # x ticks
    'ytick.labelsize': 10,    # y ticks
    'legend.fontsize': 10
})

plt.legend(markerscale=3, fontsize=7, ncol=2, frameon=True)
plt.tight_layout()
plt.show()


# sizes = df.groupby(["cluster_id","cluster_label"]).size().reset_index(name="count")
# sizes = sizes.sort_values("count", ascending=False)

# plt.figure(figsize=(10,6))
# plt.barh(range(len(sizes)), sizes["count"].values)
# plt.yticks(range(len(sizes)),
#            [f"{cid}: {lab[:40]}…" for cid, lab in sizes[["cluster_id","cluster_label"]].itertuples(index=False)])
# plt.gca().invert_yaxis()
# plt.xlabel("Number of articles")
# plt.title("Cluster sizes with labels")
# plt.tight_layout()
# plt.show()