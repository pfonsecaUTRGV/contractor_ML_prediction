import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import seaborn as sns

##### Data Preparation

# --- Load dataset ---
df = pd.read_csv("audits_english.csv")
df = df.fillna('')

# --- Encode categorical features ---
version = sklearn.__version__
major, minor, *_ = version.split(".")
minor = int(minor)

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
cat_features = encoder.fit_transform(df[['Kpi', 'contractor']])

# --- BERT embeddings for WBS ---
model = SentenceTransformer('all-MiniLM-L6-v2')
wbs_embeddings = model.encode(df['wbs'].tolist(), show_progress_bar=True)
wbs_embeddings = np.array(wbs_embeddings)

# --- PCA ---
pca = PCA(n_components=50, random_state=42)
wbs_reduced = pca.fit_transform(wbs_embeddings)
print(f"PCA reduced embeddings shape: {wbs_reduced.shape}")

# --- Combine features ---
X_cat = cat_features
X_emb = wbs_reduced
X = np.hstack([X_cat, X_emb])
y = df['Grade'].values



#Divide grades into good/bad
grade_threshold = 60
labels = np.where(df["Grade"] >= grade_threshold, "good", "bad")

# Split embeddings
good_embeddings = wbs_reduced[labels == "good"]
bad_embeddings = wbs_reduced[labels == "bad"]

# -----------------------------------------------------
# 1. Compute pairwise distances
# -----------------------------------------------------
# Distances within each group
dist_good = pairwise_distances(good_embeddings)
dist_bad = pairwise_distances(bad_embeddings)

# Extract upper triangle (to avoid duplicate distances & zeros)
intra_good_vals = dist_good[np.triu_indices_from(dist_good, k=1)]
intra_bad_vals = dist_bad[np.triu_indices_from(dist_bad, k=1)]

# Distances between groups
inter_vals = pairwise_distances(good_embeddings, bad_embeddings).flatten()

# -----------------------------------------------------
# 2. Compute averages
# -----------------------------------------------------
print("Average intra-class distance (good):", np.mean(intra_good_vals))
print("Average intra-class distance (bad):", np.mean(intra_bad_vals))
print("Average inter-class distance (good vs bad):", np.mean(inter_vals))

# -----------------------------------------------------
# 3. Statistical significance test
# -----------------------------------------------------
# Null hypothesis: inter-class distances are not larger than intra-class

# Test: compare inter-class vs combined intra-class distances
combined_intra = np.concatenate([intra_good_vals, intra_bad_vals])

t_stat, p_value = ttest_ind(inter_vals, combined_intra, equal_var=False)

print("\n--- Statistical Test ---")
print("t-statistic:", t_stat)
print("p-value:", p_value)
if p_value < 0.05:
    print("Significant difference: inter-class distances differ from intra-class.")
else:
    print("No significant difference found.")

# -----------------------------------------------------
# 4. Visualization with t-SNE
# -----------------------------------------------------
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
embeddings_2d = tsne.fit_transform(wbs_reduced)

plt.figure(figsize=(8, 6))
for grade, color in [("good", "green"), ("bad", "red")]:
    mask = (labels == grade)
    plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                label=grade, alpha=0.6, c=color)

plt.title("t-SNE visualization of WBS embeddings by grade")
plt.legend()
plt.show()



# -----------------------------------------------------
# 5. Visualization of distance distributions
# -----------------------------------------------------

# Prepare data for plotting
import pandas as pd

distance_data = pd.DataFrame({
    "Distance": np.concatenate([intra_good_vals, intra_bad_vals, inter_vals]),
    "Type": (["Intra-Good"] * len(intra_good_vals)) +
            (["Intra-Bad"] * len(intra_bad_vals)) +
            (["Inter (Good vs Bad)"] * len(inter_vals))
})

# --- Boxplot ---
plt.figure(figsize=(10, 6))
sns.boxplot(x="Type", y="Distance", data=distance_data, palette="Set2")
plt.title("Distribution of Intra- vs Inter-Class Distances (Boxplot)")
plt.xticks(rotation=15)
plt.show()

# --- Violin plot ---
plt.figure(figsize=(10, 6))
sns.violinplot(x="Type", y="Distance", data=distance_data, palette="Set2", cut=0)
plt.title("Distribution of Intra- vs Inter-Class Distances (Violin Plot)")
plt.xticks(rotation=15)
plt.show()
