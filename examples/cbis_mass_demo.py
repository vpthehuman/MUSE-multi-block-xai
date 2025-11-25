import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from muse_xai import MUSE

# Load CSVs
df_train = pd.read_csv("mass_case_description_train_set.csv")
df_test = pd.read_csv("mass_case_description_test_set.csv")
df = pd.concat([df_train, df_test], axis=0)

# Binary label: 1 = MALIGNANT, 0 = BENIGN
df["label"] = (df["pathology"] == "MALIGNANT").astype(int)

# Define blocks
clinical_block = df[["age"]].fillna(df["age"].mean())
descriptor_block = pd.get_dummies(df["margin"])

X_train, X_test, y_train, y_test = train_test_split(
    pd.concat([clinical_block, descriptor_block], axis=1),
    df["label"],
    test_size=0.2,
    random_state=42
)

blocks_train = {
    "clinical_block": X_train[["age"]],
    "descriptor_block": X_train.drop(columns="age")
}
blocks_test = {
    "clinical_block": X_test[["age"]],
    "descriptor_block": X_test.drop(columns="age")
}

# Run MUSE
muse = MUSE(random_state=42)
muse.fit(blocks_train, y_train)
metrics = muse.evaluate(blocks_test, y_test, target_names=["benign", "malignant"])
print(metrics)

muse.explain_global(blocks_train)
muse.explain_local(blocks_test, blocks_test["clinical_block"].index[0])

muse.generate_model_card(
    dataset_name="CBIS-DDSM MASS Subset",
    dataset_reference="https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM",
    task_description="Radiology + clinical metadata classification of breast lesions."
)
