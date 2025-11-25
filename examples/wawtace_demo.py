import pandas as pd
from sklearn.model_selection import train_test_split
from muse_xai import MUSE

# Load
clinical = pd.read_excel("clinical_data_wawtace_v2_15_07_2024.xlsx", index_col=0)
radiomics = pd.read_excel("radiomics_data_wawtace_09_05_2024.xlsx", index_col=0)
target = clinical["early_response"].map({"NR": 0, "R": 1})

# Split
X_clin_train, X_clin_test, y_train, y_test = train_test_split(
    clinical.drop(columns="early_response"), target, test_size=0.2, random_state=42
)
X_rad_train = radiomics.loc[X_clin_train.index]
X_rad_test = radiomics.loc[X_clin_test.index]

blocks_train = {
    "clinical_block": X_clin_train,
    "radiomics_block": X_rad_train
}
blocks_test = {
    "clinical_block": X_clin_test,
    "radiomics_block": X_rad_test
}

# Run MUSE
muse = MUSE(random_state=42)
muse.fit(blocks_train, y_train)
metrics = muse.evaluate(blocks_test, y_test, target_names=["non-responder", "responder"])
print(metrics)

muse.explain_global(blocks_train)
muse.explain_local(blocks_test, X_clin_test.index[0])

muse.generate_model_card(
    dataset_name="WAW-TACE HCC Study",
    dataset_reference="N/A (internal)",
    task_description="Predicting response to TACE therapy from clinical + radiomics blocks."
)
