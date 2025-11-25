import pandas as pd
from sklearn.model_selection import train_test_split
from muse_xai import MUSE

# Load WDBC dataset (assumes you downloaded it from UCI and named it wdbc.data)
col_names = [
    "ID", "diagnosis"
] + [f"{stat}_{feat}" for stat in ["mean", "se", "worst"]
     for feat in [
         "radius", "texture", "perimeter", "area", "smoothness",
         "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension"
     ]]

df = pd.read_csv("wdbc.data", header=None, names=col_names)
df.drop(columns=["ID"], inplace=True)

# Encode labels: B -> 0, M -> 1
df["diagnosis"] = df["diagnosis"].map({"B": 0, "M": 1})

# Split into feature blocks
block_mean = df[[c for c in df.columns if c.startswith("mean_")]]
block_se = df[[c for c in df.columns if c.startswith("se_")]]
block_worst = df[[c for c in df.columns if c.startswith("worst_")]]
y = df["diagnosis"]

# Train-test split
X_mean_train, X_mean_test, \
X_se_train, X_se_test, \
X_worst_train, X_worst_test, \
y_train, y_test = train_test_split(
    block_mean, block_se, block_worst, y, test_size=0.2, random_state=42, stratify=y
)

blocks_train = {
    "mean_block": X_mean_train,
    "se_block": X_se_train,
    "worst_block": X_worst_train,
}
blocks_test = {
    "mean_block": X_mean_test,
    "se_block": X_se_test,
    "worst_block": X_worst_test,
}

# Fit and evaluate
muse = MUSE(random_state=42)
muse.fit(blocks_train, y_train)
metrics = muse.evaluate(blocks_test, y_test, target_names=("benign", "malignant"))
print(metrics)

# Global explanations
feat_imp, block_imp = muse.explain_global(blocks_train)

# Local explanation for a sample
sample_idx = blocks_test["mean_block"].index[0]
local_exp = muse.explain_local(blocks_test, sample_idx)

# Model card
card = muse.generate_model_card(
    dataset_name="Wisconsin Diagnostic Breast Cancer (WDBC)",
    dataset_reference="https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)",
    task_description="Binary classification of breast masses (benign vs malignant)."
)
print(card)
